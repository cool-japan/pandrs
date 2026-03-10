//! Relationship graph storage and query implementation
//!
//! This module provides the core graph structure for storing and querying
//! relationships with efficient forward/reverse indexes and caching.

use super::types::{Object, Relation, RelationTuple, Subject};
use crate::error::{Error, Result};
use lru::LruCache;
use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};

/// Cache key for permission checks
type CacheKey = (String, String, String); // (subject, relation, object)

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
        // NonZeroUsize::new(1) is guaranteed to succeed since 1 is non-zero
        let cache_size =
            NonZeroUsize::new(size).unwrap_or(unsafe { NonZeroUsize::new_unchecked(1) });
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

    /// Check if subject has relation to object (with transitive resolution)
    pub fn check(&self, subject: &Subject, relation: &Relation, object: &Object) -> Result<bool> {
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

        // Perform BFS traversal to check for relationship
        let result = self.check_internal(subject, relation, object)?;

        // Update cache
        if let Ok(mut cache_guard) = self.cache.write() {
            cache_guard.put(cache_key, result);
        }

        Ok(result)
    }

    /// Internal check with BFS traversal and cycle detection
    fn check_internal(
        &self,
        subject: &Subject,
        relation: &Relation,
        object: &Object,
    ) -> Result<bool> {
        let object_key = object.to_string_format();
        let relation_name = &relation.name;

        // Check direct relationship
        if self.has_direct_relationship(subject, relation, object) {
            return Ok(true);
        }

        // BFS traversal with cycle detection
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with the target object
        queue.push_back(object_key.clone());
        visited.insert(object_key.clone());

        while let Some(current_obj) = queue.pop_front() {
            // Get all subjects with the relation to current object
            if let Some(entries) = self.forward.get(&current_obj) {
                for (subj_key, rel_key) in entries {
                    if rel_key == relation_name {
                        // Check if this subject matches or is a superset of our target subject
                        if let Ok(current_subj) = Subject::parse(subj_key) {
                            if self.subject_matches(subject, &current_subj) {
                                return Ok(true);
                            }

                            // If current subject is a subject set (has relation), expand it
                            if current_subj.relation.is_some() {
                                // This is a subject set like "team:engineering#member"
                                // Check if our subject is a member of this set
                                if self.is_member_of_set(subject, &current_subj)? {
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
            }

            // Traverse parent relationships (for hierarchical permissions)
            if let Some(parent_entries) = self.forward.get(&current_obj) {
                for (_, rel_key) in parent_entries {
                    if rel_key == "parent" {
                        // Look for parent objects to traverse up the hierarchy
                        if let Some(reverse_entries) = self.reverse.get(&current_obj) {
                            for (parent_rel, parent_obj) in reverse_entries {
                                if parent_rel == "parent" && !visited.contains(parent_obj) {
                                    visited.insert(parent_obj.clone());
                                    queue.push_back(parent_obj.clone());
                                }
                            }
                        }
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

    /// Check if subject is a member of a subject set
    fn is_member_of_set(&self, subject: &Subject, set: &Subject) -> Result<bool> {
        // Extract the relation from the set
        if let Some(ref set_relation) = set.relation {
            // Create a relation to check
            let rel = Relation::new(set_relation);
            let obj = Object::new(&set.subject_type, &set.subject_id);

            // Recursively check if subject has this relationship
            self.check(subject, &rel, &obj)
        } else {
            Ok(false)
        }
    }

    /// Expand all subjects with relation to object
    pub fn expand(&self, relation: &Relation, object: &Object) -> Result<Vec<Subject>> {
        let object_key = object.to_string_format();
        let relation_name = &relation.name;
        let mut subjects = Vec::new();

        if let Some(entries) = self.forward.get(&object_key) {
            for (subj_key, rel_key) in entries {
                if rel_key == relation_name {
                    if let Ok(subject) = Subject::parse(subj_key) {
                        subjects.push(subject);
                    }
                }
            }
        }

        Ok(subjects)
    }

    /// List all objects subject has relation to
    pub fn list_objects(&self, subject: &Subject, relation: &Relation) -> Result<Vec<Object>> {
        let subject_key = subject.to_string_format();
        let relation_name = &relation.name;
        let mut objects = Vec::new();

        if let Some(entries) = self.reverse.get(&subject_key) {
            for (rel, obj_key) in entries {
                if rel == relation_name {
                    if let Ok(object) = Object::parse(obj_key) {
                        objects.push(object);
                    }
                }
            }
        }

        Ok(objects)
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
            )
            .expect("check should succeed");
        assert!(has_rel);

        // Different subject should not have relationship
        let has_rel = graph
            .check(
                &Subject::new("user", "bob"),
                &Relation::new("owner"),
                &Object::new("document", "123"),
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
            )
            .expect("check should succeed");
        assert!(result1);

        // Second check (cache hit)
        let result2 = graph_mut
            .check(
                &Subject::new("user", "alice"),
                &Relation::new("owner"),
                &Object::new("document", "123"),
            )
            .expect("check should succeed");
        assert!(result2);

        let (cache_len, cache_cap) = graph_mut.cache_stats().expect("stats should succeed");
        assert_eq!(cache_len, 1);
        assert_eq!(cache_cap, 10);
    }
}
