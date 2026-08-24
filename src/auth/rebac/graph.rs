//! Relationship graph storage and query implementation
//!
//! This module provides the core graph structure for storing and querying
//! relationships with efficient forward/reverse indexes and caching.

use super::schema::{PermissionSchema, RewriteRule};
use super::types::{Object, Relation, RelationTuple, Subject};
use crate::error::{Error, Result};
use lru::LruCache;
use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};

/// Cache key for permission checks
type CacheKey = (String, String, String); // (subject, relation, object)

/// A `(subject, relation, object)` triple currently being resolved. Threaded
/// through the recursive evaluator as the cycle-detection path set.
type EvalKey = (String, String, String);

/// Hard ceiling on recursive rewrite / subject-set resolution depth.
///
/// The path set below already makes cyclic grants terminate; this is a second,
/// unconditional guard so that even a non-cyclic but pathologically deep schema
/// cannot exhaust the stack. 32 is far beyond any realistic relation-nesting
/// depth while remaining cheap.
const MAX_RESOLUTION_DEPTH: usize = 32;

/// Relationship graph with forward/reverse indexes
#[derive(Debug)]
pub struct RelationshipGraph {
    /// Forward index: object -> [(subject, relation)]
    forward: HashMap<String, Vec<(String, String)>>,
    /// Reverse index: subject -> [(relation, object)]
    reverse: HashMap<String, Vec<(String, String)>>,
    /// All tuples stored
    tuples: HashSet<RelationTuple>,
    /// Cache for computed permissions (thread-safe)
    cache: Arc<RwLock<LruCache<CacheKey, bool>>>,
    /// Maximum cache size
    cache_size: usize,
}

impl RelationshipGraph {
    /// Create a new relationship graph
    pub fn new() -> Self {
        Self::with_cache_size(1000)
    }

    /// Create a new relationship graph with specified cache size
    pub fn with_cache_size(size: usize) -> Self {
        // A zero cache size is meaningless; fall back to the smallest non-zero
        // capacity. `NonZeroUsize::MIN` avoids the previous gratuitous `unsafe`.
        let cache_size = NonZeroUsize::new(size).unwrap_or(NonZeroUsize::MIN);
        RelationshipGraph {
            forward: HashMap::new(),
            reverse: HashMap::new(),
            tuples: HashSet::new(),
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            cache_size: size,
        }
    }

    /// Add a relationship tuple
    pub fn add_tuple(&mut self, tuple: RelationTuple) -> Result<()> {
        // Check if tuple already exists
        if self.tuples.contains(&tuple) {
            return Ok(()); // Already exists, no-op
        }

        let subject_key = tuple.subject.to_string_format();
        let relation_key = tuple.relation.name.clone();
        let object_key = tuple.object.to_string_format();

        // Add to forward index: object -> (subject, relation)
        self.forward
            .entry(object_key.clone())
            .or_insert_with(Vec::new)
            .push((subject_key.clone(), relation_key.clone()));

        // Add to reverse index: subject -> (relation, object)
        self.reverse
            .entry(subject_key)
            .or_insert_with(Vec::new)
            .push((relation_key, object_key));

        // Store the tuple
        self.tuples.insert(tuple);

        // Invalidate cache
        self.clear_cache();

        Ok(())
    }

    /// Remove a relationship tuple
    pub fn remove_tuple(&mut self, tuple: &RelationTuple) -> Result<()> {
        if !self.tuples.contains(tuple) {
            return Err(Error::InvalidInput("Tuple not found".to_string()));
        }

        let subject_key = tuple.subject.to_string_format();
        let relation_key = &tuple.relation.name;
        let object_key = tuple.object.to_string_format();

        // Remove from forward index
        if let Some(entries) = self.forward.get_mut(&object_key) {
            entries.retain(|(s, r)| s != &subject_key || r != relation_key);
            if entries.is_empty() {
                self.forward.remove(&object_key);
            }
        }

        // Remove from reverse index
        if let Some(entries) = self.reverse.get_mut(&subject_key) {
            entries.retain(|(r, o)| r != relation_key || o != &object_key);
            if entries.is_empty() {
                self.reverse.remove(&subject_key);
            }
        }

        // Remove the tuple
        self.tuples.remove(tuple);

        // Invalidate cache
        self.clear_cache();

        Ok(())
    }

    /// Whether an exact tuple is present. Cheap membership test used by the
    /// manager to make revocation idempotent (revoking an absent grant is a
    /// success, not an error).
    pub fn contains_tuple(&self, tuple: &RelationTuple) -> bool {
        self.tuples.contains(tuple)
    }

    /// Check if subject has relation to object (with transitive resolution).
    ///
    /// `schema` supplies the optional rewrite rules (Zanzibar-style
    /// `ComputedUserset` / `TupleToUserset` / `Union` / `Intersection` /
    /// `Exclusion`). When the schema has no entry for `(object type, relation)`
    /// the check falls back to *primitive* resolution (direct tuple +
    /// subject-set membership + hierarchical `parent` inheritance), which is
    /// the historical behaviour. Callers that want the flat model pass an empty
    /// schema.
    pub fn check(
        &self,
        subject: &Subject,
        relation: &Relation,
        object: &Object,
        schema: &PermissionSchema,
    ) -> Result<bool> {
        let cache_key = (
            subject.to_string_format(),
            relation.name.clone(),
            object.to_string_format(),
        );

        // Check cache first
        if let Ok(cache_guard) = self.cache.read() {
            if let Some(&result) = cache_guard.peek(&cache_key) {
                return Ok(result);
            }
        }

        // Resolve from an empty path set at depth 0. Only the top-level result
        // is cached; intermediate results computed under an active
        // cycle-guard are never cached (they may be conservatively `false`
        // because a back-edge was cut).
        let mut path: HashSet<EvalKey> = HashSet::new();
        let result = self.evaluate(subject, relation, object, schema, &mut path, 0)?;

        // Update cache
        if let Ok(mut cache_guard) = self.cache.write() {
            cache_guard.put(cache_key, result);
        }

        Ok(result)
    }

    /// Recursive relation resolver with cycle detection and depth limiting.
    ///
    /// `path` holds every `(subject, relation, object)` triple on the current
    /// resolution stack. Re-encountering a triple is a cycle and yields
    /// `Ok(false)` rather than recursing forever. The triple is removed on exit
    /// (DFS back-tracking) so that shared sub-goals reached along *different*
    /// branches are still evaluated on their own merits.
    fn evaluate(
        &self,
        subject: &Subject,
        relation: &Relation,
        object: &Object,
        schema: &PermissionSchema,
        path: &mut HashSet<EvalKey>,
        depth: usize,
    ) -> Result<bool> {
        if depth > MAX_RESOLUTION_DEPTH {
            return Ok(false);
        }

        let key: EvalKey = (
            subject.to_string_format(),
            relation.name.clone(),
            object.to_string_format(),
        );
        if path.contains(&key) {
            // Cyclic grant (e.g. two subject-sets that reference each other).
            // Terminate instead of overflowing the stack.
            return Ok(false);
        }
        path.insert(key.clone());

        let rule = schema
            .get_type(&object.object_type)
            .and_then(|t| t.get_relation(&relation.name))
            .map(|def| def.rewrite.clone());

        let result = match rule {
            Some(rule) => {
                self.evaluate_rule(subject, &relation.name, object, &rule, schema, path, depth)
            }
            // No schema entry for this (type, relation): primitive resolution.
            None => self.primitive_check(subject, relation, object, schema, path, depth),
        };

        path.remove(&key);
        result
    }

    /// Evaluate a single [`RewriteRule`] in the context of `relation_name` on
    /// `object`.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_rule(
        &self,
        subject: &Subject,
        relation_name: &str,
        object: &Object,
        rule: &RewriteRule,
        schema: &PermissionSchema,
        path: &mut HashSet<EvalKey>,
        depth: usize,
    ) -> Result<bool> {
        match rule {
            RewriteRule::This => {
                let relation = Relation::new(relation_name);
                self.primitive_check(subject, &relation, object, schema, path, depth)
            }
            RewriteRule::ComputedUserset { relation } => {
                let rel = Relation::new(relation);
                self.evaluate(subject, &rel, object, schema, path, depth + 1)
            }
            RewriteRule::TupleToUserset {
                tupleset_relation,
                computed_relation,
            } => {
                // Follow `tupleset_relation` outward from `object` (object acts
                // as a subject in the reverse index), then check
                // `computed_relation` on each target.
                let object_key = object.to_string_format();
                if let Some(entries) = self.reverse.get(&object_key) {
                    for (rel, target) in entries {
                        if rel == tupleset_relation {
                            if let Ok(target_obj) = Object::parse(target) {
                                let comp = Relation::new(computed_relation);
                                if self.evaluate(
                                    subject,
                                    &comp,
                                    &target_obj,
                                    schema,
                                    path,
                                    depth + 1,
                                )? {
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
                Ok(false)
            }
            RewriteRule::Union(rules) => {
                for r in rules {
                    if self.evaluate_rule(subject, relation_name, object, r, schema, path, depth)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            RewriteRule::Intersection(rules) => {
                for r in rules {
                    if !self.evaluate_rule(
                        subject,
                        relation_name,
                        object,
                        r,
                        schema,
                        path,
                        depth,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            RewriteRule::Exclusion { base, subtract } => {
                let has_base =
                    self.evaluate_rule(subject, relation_name, object, base, schema, path, depth)?;
                if !has_base {
                    return Ok(false);
                }
                let excluded = self.evaluate_rule(
                    subject,
                    relation_name,
                    object,
                    subtract,
                    schema,
                    path,
                    depth,
                )?;
                Ok(!excluded)
            }
        }
    }

    /// Primitive relation resolution: direct tuple, subject-set membership, and
    /// hierarchical `parent` inheritance for a *single* concrete relation.
    ///
    /// This is the leaf of schema evaluation (`RewriteRule::This`) and the
    /// fallback when no schema entry exists.
    fn primitive_check(
        &self,
        subject: &Subject,
        relation: &Relation,
        object: &Object,
        schema: &PermissionSchema,
        path: &mut HashSet<EvalKey>,
        depth: usize,
    ) -> Result<bool> {
        if depth > MAX_RESOLUTION_DEPTH {
            return Ok(false);
        }

        let object_key = object.to_string_format();
        let relation_name = &relation.name;

        // Direct relationship on the target object.
        if self.has_direct_relationship(subject, relation, object) {
            return Ok(true);
        }

        // BFS *up* the hierarchy: the target object and every ancestor reached
        // by following `parent` edges. `visited` bounds the walk even when the
        // `parent` graph contains a cycle.
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(object_key.clone());
        visited.insert(object_key);

        while let Some(current_obj) = queue.pop_front() {
            // Any subject granted `relation` directly on this object/ancestor.
            if let Some(entries) = self.forward.get(&current_obj) {
                for (subj_key, rel_key) in entries {
                    if rel_key == relation_name {
                        if let Ok(current_subj) = Subject::parse(subj_key) {
                            if self.subject_matches(subject, &current_subj) {
                                return Ok(true);
                            }
                            // Subject set (e.g. "team:engineering#member"):
                            // is our concrete subject a member of it?
                            if current_subj.relation.is_some()
                                && self.is_member_of_set(
                                    subject,
                                    &current_subj,
                                    schema,
                                    path,
                                    depth,
                                )?
                            {
                                return Ok(true);
                            }
                        }
                    }
                }
            }

            // Enqueue parents. A tuple `(current_obj, "parent", parent_obj)` is
            // stored in the *reverse* index as `current_obj -> ("parent",
            // parent_obj)`. The previous implementation consulted the forward
            // index here, so folder inheritance never fired.
            if let Some(reverse_entries) = self.reverse.get(&current_obj) {
                for (rel, parent_obj) in reverse_entries {
                    if rel == "parent" && !visited.contains(parent_obj) {
                        visited.insert(parent_obj.clone());
                        queue.push_back(parent_obj.clone());
                    }
                }
            }
        }

        Ok(false)
    }

    /// Check if there's a direct relationship
    fn has_direct_relationship(
        &self,
        subject: &Subject,
        relation: &Relation,
        object: &Object,
    ) -> bool {
        let subject_key = subject.to_string_format();
        let relation_name = &relation.name;
        let object_key = object.to_string_format();

        if let Some(entries) = self.reverse.get(&subject_key) {
            for (rel, obj) in entries {
                if rel == relation_name && obj == &object_key {
                    return true;
                }
            }
        }
        false
    }

    /// Check if subject matches another subject (considering subject sets)
    fn subject_matches(&self, target: &Subject, candidate: &Subject) -> bool {
        if target.subject_type != candidate.subject_type {
            return false;
        }

        if target.subject_id != candidate.subject_id {
            return false;
        }

        // Both must have the same relation (or both None)
        target.relation == candidate.relation
    }

    /// Check if subject is a member of a subject set.
    ///
    /// Crucially this re-enters the *guarded* [`evaluate`] with the shared
    /// `path` set and an incremented depth, rather than the previous
    /// `self.check` which started a fresh cycle guard on every hop. Two
    /// subject-sets that reference each other (`team:a#member -> team:b`,
    /// `team:b#member -> team:a`) therefore terminate with `Ok(false)` instead
    /// of recursing until the stack overflows (an uncatchable SIGABRT).
    fn is_member_of_set(
        &self,
        subject: &Subject,
        set: &Subject,
        schema: &PermissionSchema,
        path: &mut HashSet<EvalKey>,
        depth: usize,
    ) -> Result<bool> {
        if let Some(ref set_relation) = set.relation {
            let rel = Relation::new(set_relation);
            let obj = Object::new(&set.subject_type, &set.subject_id);
            self.evaluate(subject, &rel, &obj, schema, path, depth + 1)
        } else {
            Ok(false)
        }
    }

    /// Expand all subjects that have `relation` on `object`, following the same
    /// transitive rules the check uses: hierarchical `parent` inheritance and
    /// subject-set membership. Returns the concrete (principal) subjects plus
    /// any subject-set tokens encountered, so an access review sees everyone
    /// `check` would authorize rather than only the directly-attached tuples.
    pub fn expand(&self, relation: &Relation, object: &Object) -> Result<Vec<Subject>> {
        let mut found: HashSet<String> = HashSet::new();
        let mut visited: HashSet<String> = HashSet::new();
        self.collect_subjects(
            &object.to_string_format(),
            &relation.name,
            &mut found,
            &mut visited,
            0,
        );

        let mut subjects: Vec<Subject> = found
            .into_iter()
            .filter_map(|s| Subject::parse(&s).ok())
            .collect();
        // Deterministic ordering for callers/tests.
        subjects.sort_by(|a, b| a.to_string_format().cmp(&b.to_string_format()));
        Ok(subjects)
    }

    /// Recursive helper for [`expand`]: gather every subject with `relation` on
    /// `object_key`, expanding subject sets into their members and walking down
    /// into child objects (which inherit the relation from this parent).
    fn collect_subjects(
        &self,
        object_key: &str,
        relation_name: &str,
        found: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        depth: usize,
    ) {
        if depth > MAX_RESOLUTION_DEPTH || !visited.insert(object_key.to_string()) {
            return;
        }

        if let Some(entries) = self.forward.get(object_key) {
            for (subj_key, rel_key) in entries {
                if rel_key == relation_name {
                    if let Ok(subj) = Subject::parse(subj_key) {
                        if let Some(set_rel) = subj.relation.clone() {
                            // Subject set: record the token and expand members.
                            found.insert(subj_key.clone());
                            let set_obj = format!("{}:{}", subj.subject_type, subj.subject_id);
                            self.collect_subjects(&set_obj, &set_rel, found, visited, depth + 1);
                        } else {
                            found.insert(subj_key.clone());
                        }
                    }
                }
                // Children of this object inherit the relation (hierarchy):
                // a tuple `(child, "parent", object)` is stored forward as
                // `object -> (child, "parent")`.
                if rel_key == "parent" {
                    let child_key = subj_key.clone();
                    self.collect_subjects(&child_key, relation_name, found, visited, depth + 1);
                }
            }
        }
    }

    /// List all objects `subject` has `relation` to, transitively: direct
    /// grants, objects inherited through hierarchical `parent` descendants, and
    /// objects granted to any subject set the subject belongs to. Sharing the
    /// traversal with `check` prevents access reviews from under-reporting.
    pub fn list_objects(&self, subject: &Subject, relation: &Relation) -> Result<Vec<Object>> {
        let mut found: HashSet<String> = HashSet::new();
        let mut visited: HashSet<(String, String)> = HashSet::new();
        self.collect_objects(
            &subject.to_string_format(),
            &relation.name,
            &mut found,
            &mut visited,
            0,
        );

        let mut objects: Vec<Object> = found
            .into_iter()
            .filter_map(|o| Object::parse(&o).ok())
            .collect();
        objects.sort_by(|a, b| a.to_string_format().cmp(&b.to_string_format()));
        Ok(objects)
    }

    /// Recursive helper for [`list_objects`].
    fn collect_objects(
        &self,
        subject_key: &str,
        relation_name: &str,
        found: &mut HashSet<String>,
        visited: &mut HashSet<(String, String)>,
        depth: usize,
    ) {
        if depth > MAX_RESOLUTION_DEPTH
            || !visited.insert((subject_key.to_string(), relation_name.to_string()))
        {
            return;
        }

        if let Some(entries) = self.reverse.get(subject_key) {
            // Snapshot to avoid holding the borrow across recursion.
            let entries: Vec<(String, String)> = entries.clone();
            for (rel, obj_key) in &entries {
                if rel == relation_name {
                    if found.insert(obj_key.clone()) {
                        // Descend into children that inherit through `parent`.
                        self.collect_descendants(obj_key, relation_name, found, visited, depth + 1);
                    }
                }
            }

            // Objects granted to a subject set this subject belongs to. For a
            // membership tuple `(subject, memberRel, setObj)`, the set token is
            // `setObj#memberRel`; anything granted `relation` to that token is
            // inherited by the subject.
            for (member_rel, set_obj) in &entries {
                let set_token = format!("{}#{}", set_obj, member_rel);
                self.collect_objects(&set_token, relation_name, found, visited, depth + 1);
            }
        }
    }

    /// Collect descendant objects that inherit `relation` from `parent_key`
    /// through `parent` edges (`object -> (child, "parent")` in the forward
    /// index).
    fn collect_descendants(
        &self,
        parent_key: &str,
        relation_name: &str,
        found: &mut HashSet<String>,
        visited: &mut HashSet<(String, String)>,
        depth: usize,
    ) {
        if depth > MAX_RESOLUTION_DEPTH {
            return;
        }
        if let Some(entries) = self.forward.get(parent_key) {
            let children: Vec<String> = entries
                .iter()
                .filter(|(_, rel)| rel == "parent")
                .map(|(child, _)| child.clone())
                .collect();
            for child in children {
                if found.insert(child.clone()) {
                    self.collect_descendants(&child, relation_name, found, visited, depth + 1);
                }
            }
        }
    }

    /// Get all tuples
    pub fn get_all_tuples(&self) -> Vec<RelationTuple> {
        self.tuples.iter().cloned().collect()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache_guard) = self.cache.write() {
            cache_guard.clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<(usize, usize)> {
        if let Ok(cache_guard) = self.cache.read() {
            Ok((cache_guard.len(), self.cache_size))
        } else {
            Err(Error::InvalidOperation(
                "Failed to acquire cache lock".to_string(),
            ))
        }
    }
}

impl Default for RelationshipGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::rebac::schema::PermissionSchema;

    #[test]
    fn test_add_and_check_tuple() {
        let mut graph = RelationshipGraph::new();

        let tuple = RelationTuple::new(
            Subject::new("user", "alice"),
            Relation::new("owner"),
            Object::new("document", "123"),
        );

        graph.add_tuple(tuple.clone()).expect("add should succeed");

        // Direct relationship should exist
        let has_rel = graph
            .check(
                &Subject::new("user", "alice"),
                &Relation::new("owner"),
                &Object::new("document", "123"),
                &PermissionSchema::new(),
            )
            .expect("check should succeed");
        assert!(has_rel);

        // Different subject should not have relationship
        let has_rel = graph
            .check(
                &Subject::new("user", "bob"),
                &Relation::new("owner"),
                &Object::new("document", "123"),
                &PermissionSchema::new(),
            )
            .expect("check should succeed");
        assert!(!has_rel);
    }

    #[test]
    fn test_remove_tuple() {
        let mut graph = RelationshipGraph::new();

        let tuple = RelationTuple::new(
            Subject::new("user", "alice"),
            Relation::new("owner"),
            Object::new("document", "123"),
        );

        graph.add_tuple(tuple.clone()).expect("add should succeed");
        graph.remove_tuple(&tuple).expect("remove should succeed");

        let has_rel = graph
            .check(
                &Subject::new("user", "alice"),
                &Relation::new("owner"),
                &Object::new("document", "123"),
                &PermissionSchema::new(),
            )
            .expect("check should succeed");
        assert!(!has_rel);
    }

    #[test]
    fn test_expand_subjects() {
        let mut graph = RelationshipGraph::new();

        graph
            .add_tuple(RelationTuple::new(
                Subject::new("user", "alice"),
                Relation::new("viewer"),
                Object::new("document", "123"),
            ))
            .expect("add should succeed");

        graph
            .add_tuple(RelationTuple::new(
                Subject::new("user", "bob"),
                Relation::new("viewer"),
                Object::new("document", "123"),
            ))
            .expect("add should succeed");

        let subjects = graph
            .expand(&Relation::new("viewer"), &Object::new("document", "123"))
            .expect("expand should succeed");

        assert_eq!(subjects.len(), 2);
    }

    #[test]
    fn test_list_objects() {
        let mut graph = RelationshipGraph::new();

        graph
            .add_tuple(RelationTuple::new(
                Subject::new("user", "alice"),
                Relation::new("owner"),
                Object::new("document", "123"),
            ))
            .expect("add should succeed");

        graph
            .add_tuple(RelationTuple::new(
                Subject::new("user", "alice"),
                Relation::new("owner"),
                Object::new("document", "456"),
            ))
            .expect("add should succeed");

        let objects = graph
            .list_objects(&Subject::new("user", "alice"), &Relation::new("owner"))
            .expect("list should succeed");

        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn test_cache() {
        let graph = RelationshipGraph::with_cache_size(10);

        // Add a tuple
        let mut graph_mut = graph;
        graph_mut
            .add_tuple(RelationTuple::new(
                Subject::new("user", "alice"),
                Relation::new("owner"),
                Object::new("document", "123"),
            ))
            .expect("add should succeed");

        // First check (cache miss)
        let result1 = graph_mut
            .check(
                &Subject::new("user", "alice"),
                &Relation::new("owner"),
                &Object::new("document", "123"),
                &PermissionSchema::new(),
            )
            .expect("check should succeed");
        assert!(result1);

        // Second check (cache hit)
        let result2 = graph_mut
            .check(
                &Subject::new("user", "alice"),
                &Relation::new("owner"),
                &Object::new("document", "123"),
                &PermissionSchema::new(),
            )
            .expect("check should succeed");
        assert!(result2);

        let (cache_len, cache_cap) = graph_mut.cache_stats().expect("stats should succeed");
        assert_eq!(cache_len, 1);
        assert_eq!(cache_cap, 10);
    }
}
