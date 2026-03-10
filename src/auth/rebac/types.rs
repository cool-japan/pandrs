//! Core types for ReBAC (Relationship-Based Access Control)
//!
//! This module defines the fundamental types used in the ReBAC system,
//! including subjects, relations, objects, and relationship tuples.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Subject represents an entity that can have relationships
/// Examples: "user:alice", "team:engineering#member", "group:admins"
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Subject {
    /// Subject type (e.g., "user", "team", "group")
    pub subject_type: String,
    /// Subject ID (e.g., "alice", "engineering")
    pub subject_id: String,
    /// Optional relation for subject sets (e.g., "member" in "team:engineering#member")
    pub relation: Option<String>,
}

impl Subject {
    /// Create a new subject
    pub fn new(subject_type: impl Into<String>, subject_id: impl Into<String>) -> Self {
        Subject {
            subject_type: subject_type.into(),
            subject_id: subject_id.into(),
            relation: None,
        }
    }

    /// Create a subject with a relation (subject set)
    pub fn with_relation(
        subject_type: impl Into<String>,
        subject_id: impl Into<String>,
        relation: impl Into<String>,
    ) -> Self {
        Subject {
            subject_type: subject_type.into(),
            subject_id: subject_id.into(),
            relation: Some(relation.into()),
        }
    }

    /// Parse from string format: "type:id" or "type:id#relation"
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid subject format: {}", s));
        }

        let subject_type = parts[0].to_string();
        let id_and_relation = parts[1];

        if id_and_relation.contains('#') {
            let rel_parts: Vec<&str> = id_and_relation.split('#').collect();
            if rel_parts.len() != 2 {
                return Err(format!("Invalid subject relation format: {}", s));
            }
            Ok(Subject {
                subject_type,
                subject_id: rel_parts[0].to_string(),
                relation: Some(rel_parts[1].to_string()),
            })
        } else {
            Ok(Subject {
                subject_type,
                subject_id: id_and_relation.to_string(),
                relation: None,
            })
        }
    }

    /// Convert to string format
    pub fn to_string_format(&self) -> String {
        if let Some(ref rel) = self.relation {
            format!("{}:{}#{}", self.subject_type, self.subject_id, rel)
        } else {
            format!("{}:{}", self.subject_type, self.subject_id)
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_format())
    }
}

/// Relation represents a type of relationship between subjects and objects
/// Examples: "owner", "editor", "viewer", "member", "parent"
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Relation {
    /// Relation name
    pub name: String,
}

impl Relation {
    /// Create a new relation
    pub fn new(name: impl Into<String>) -> Self {
        Relation { name: name.into() }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Result<Self, String> {
        if s.is_empty() {
            return Err("Relation cannot be empty".to_string());
        }
        Ok(Relation {
            name: s.to_string(),
        })
    }
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Object represents a resource that can be accessed
/// Examples: "document:123", "folder:456", "tenant:acme"
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Object {
    /// Object type (e.g., "document", "folder", "tenant")
    pub object_type: String,
    /// Object ID (e.g., "123", "456", "acme")
    pub object_id: String,
}

impl Object {
    /// Create a new object
    pub fn new(object_type: impl Into<String>, object_id: impl Into<String>) -> Self {
        Object {
            object_type: object_type.into(),
            object_id: object_id.into(),
        }
    }

    /// Parse from string format: "type:id"
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid object format: {}", s));
        }

        Ok(Object {
            object_type: parts[0].to_string(),
            object_id: parts[1].to_string(),
        })
    }

    /// Convert to string format
    pub fn to_string_format(&self) -> String {
        format!("{}:{}", self.object_type, self.object_id)
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_format())
    }
}

/// RelationTuple represents a relationship: (subject, relation, object)
/// Example: ("user:alice", "owner", "document:123")
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelationTuple {
    /// The subject of the relationship
    pub subject: Subject,
    /// The relation type
    pub relation: Relation,
    /// The object of the relationship
    pub object: Object,
}

impl RelationTuple {
    /// Create a new relation tuple
    pub fn new(subject: Subject, relation: Relation, object: Object) -> Self {
        RelationTuple {
            subject,
            relation,
            object,
        }
    }

    /// Parse from string format: "subject relation object"
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(format!("Invalid tuple format: {}", s));
        }

        Ok(RelationTuple {
            subject: Subject::parse(parts[0])?,
            relation: Relation::parse(parts[1])?,
            object: Object::parse(parts[2])?,
        })
    }

    /// Convert to string format
    pub fn to_string_format(&self) -> String {
        format!(
            "{} {} {}",
            self.subject.to_string_format(),
            self.relation.name,
            self.object.to_string_format()
        )
    }
}

impl fmt::Display for RelationTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_format())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subject_parse() {
        let subject = Subject::parse("user:alice").expect("parse should succeed");
        assert_eq!(subject.subject_type, "user");
        assert_eq!(subject.subject_id, "alice");
        assert_eq!(subject.relation, None);

        let subject_with_rel =
            Subject::parse("team:engineering#member").expect("parse should succeed");
        assert_eq!(subject_with_rel.subject_type, "team");
        assert_eq!(subject_with_rel.subject_id, "engineering");
        assert_eq!(subject_with_rel.relation, Some("member".to_string()));
    }

    #[test]
    fn test_subject_to_string() {
        let subject = Subject::new("user", "alice");
        assert_eq!(subject.to_string_format(), "user:alice");

        let subject_with_rel = Subject::with_relation("team", "engineering", "member");
        assert_eq!(
            subject_with_rel.to_string_format(),
            "team:engineering#member"
        );
    }

    #[test]
    fn test_object_parse() {
        let object = Object::parse("document:123").expect("parse should succeed");
        assert_eq!(object.object_type, "document");
        assert_eq!(object.object_id, "123");
    }

    #[test]
    fn test_relation_tuple_parse() {
        let tuple =
            RelationTuple::parse("user:alice owner document:123").expect("parse should succeed");
        assert_eq!(tuple.subject.subject_id, "alice");
        assert_eq!(tuple.relation.name, "owner");
        assert_eq!(tuple.object.object_id, "123");
    }

    #[test]
    fn test_relation_tuple_to_string() {
        let tuple = RelationTuple::new(
            Subject::new("user", "alice"),
            Relation::new("owner"),
            Object::new("document", "123"),
        );
        assert_eq!(tuple.to_string_format(), "user:alice owner document:123");
    }
}
