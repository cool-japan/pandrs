//! Permission schema definitions for ReBAC
//!
//! This module provides schema definitions that describe how permissions
//! are computed from relationships (similar to Zanzibar, Auth0 FGA).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Permission schema defining how relations work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSchema {
    /// Type definitions (e.g., "document", "folder")
    pub types: HashMap<String, TypeDefinition>,
}

impl PermissionSchema {
    /// Create a new empty schema
    pub fn new() -> Self {
        PermissionSchema {
            types: HashMap::new(),
        }
    }

    /// Add a type definition
    pub fn add_type(&mut self, name: impl Into<String>, type_def: TypeDefinition) {
        self.types.insert(name.into(), type_def);
    }

    /// Get a type definition
    pub fn get_type(&self, name: &str) -> Option<&TypeDefinition> {
        self.types.get(name)
    }

    /// Create a default schema for common patterns
    pub fn default_schema() -> Self {
        let mut schema = Self::new();

        // Document type
        let mut document_type = TypeDefinition::new();
        document_type.add_relation(
            "owner",
            RelationDefinition {
                description: Some("Direct owner of the document".to_string()),
                rewrite: RewriteRule::This,
            },
        );
        document_type.add_relation(
            "editor",
            RelationDefinition {
                description: Some("Can edit the document (includes owners)".to_string()),
                rewrite: RewriteRule::Union(vec![
                    RewriteRule::This,
                    RewriteRule::ComputedUserset {
                        relation: "owner".to_string(),
                    },
                ]),
            },
        );
        document_type.add_relation(
            "viewer",
            RelationDefinition {
                description: Some("Can view the document (includes editors)".to_string()),
                rewrite: RewriteRule::Union(vec![
                    RewriteRule::This,
                    RewriteRule::ComputedUserset {
                        relation: "editor".to_string(),
                    },
                ]),
            },
        );
        schema.add_type("document", document_type);

        // Folder type
        let mut folder_type = TypeDefinition::new();
        folder_type.add_relation(
            "owner",
            RelationDefinition {
                description: Some("Direct owner of the folder".to_string()),
                rewrite: RewriteRule::This,
            },
        );
        folder_type.add_relation(
            "viewer",
            RelationDefinition {
                description: Some("Can view the folder".to_string()),
                rewrite: RewriteRule::Union(vec![
                    RewriteRule::This,
                    RewriteRule::ComputedUserset {
                        relation: "owner".to_string(),
                    },
                ]),
            },
        );
        schema.add_type("folder", folder_type);

        // Team type
        let mut team_type = TypeDefinition::new();
        team_type.add_relation(
            "member",
            RelationDefinition {
                description: Some("Member of the team".to_string()),
                rewrite: RewriteRule::This,
            },
        );
        team_type.add_relation(
            "admin",
            RelationDefinition {
                description: Some("Administrator of the team".to_string()),
                rewrite: RewriteRule::This,
            },
        );
        schema.add_type("team", team_type);

        // Tenant type
        let mut tenant_type = TypeDefinition::new();
        tenant_type.add_relation(
            "admin",
            RelationDefinition {
                description: Some("Administrator of the tenant".to_string()),
                rewrite: RewriteRule::This,
            },
        );
        tenant_type.add_relation(
            "manager",
            RelationDefinition {
                description: Some("Manager in the tenant".to_string()),
                rewrite: RewriteRule::This,
            },
        );
        tenant_type.add_relation(
            "member",
            RelationDefinition {
                description: Some("Member of the tenant".to_string()),
                rewrite: RewriteRule::Union(vec![
                    RewriteRule::This,
                    RewriteRule::ComputedUserset {
                        relation: "admin".to_string(),
                    },
                    RewriteRule::ComputedUserset {
                        relation: "manager".to_string(),
                    },
                ]),
            },
        );
        schema.add_type("tenant", tenant_type);

        schema
    }
}

impl Default for PermissionSchema {
    fn default() -> Self {
        Self::default_schema()
    }
}

/// Type definition in the schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinition {
    /// Relations defined for this type
    pub relations: HashMap<String, RelationDefinition>,
}

impl TypeDefinition {
    /// Create a new type definition
    pub fn new() -> Self {
        TypeDefinition {
            relations: HashMap::new(),
        }
    }

    /// Add a relation definition
    pub fn add_relation(&mut self, name: impl Into<String>, relation: RelationDefinition) {
        self.relations.insert(name.into(), relation);
    }

    /// Get a relation definition
    pub fn get_relation(&self, name: &str) -> Option<&RelationDefinition> {
        self.relations.get(name)
    }
}

impl Default for TypeDefinition {
    fn default() -> Self {
        Self::new()
    }
}

/// Relation definition describing how a relation is computed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationDefinition {
    /// Description of this relation
    pub description: Option<String>,
    /// Rewrite rule for computing this relation
    pub rewrite: RewriteRule,
}

/// Rewrite rules define how permissions are computed
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RewriteRule {
    /// Direct relation (this)
    This,

    /// Computed userset: derive from another relation
    ComputedUserset {
        /// Relation name to compute from
        relation: String,
    },

    /// Tuple to userset: follow a relation and then check another relation
    TupleToUserset {
        /// Tupleset relation to follow
        tupleset_relation: String,
        /// Computed userset relation to check
        computed_relation: String,
    },

    /// Union: subject has permission if ANY of the rules match
    Union(Vec<RewriteRule>),

    /// Intersection: subject has permission if ALL of the rules match
    Intersection(Vec<RewriteRule>),

    /// Exclusion: subject has permission from base but not in exclusion
    Exclusion {
        /// Base rule
        base: Box<RewriteRule>,
        /// Exclusion rule
        subtract: Box<RewriteRule>,
    },
}

impl RewriteRule {
    /// Check if this is a simple "This" rule
    pub fn is_this(&self) -> bool {
        matches!(self, RewriteRule::This)
    }

    /// Check if this is a union rule
    pub fn is_union(&self) -> bool {
        matches!(self, RewriteRule::Union(_))
    }

    /// Check if this is a computed userset
    pub fn is_computed_userset(&self) -> bool {
        matches!(self, RewriteRule::ComputedUserset { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let schema = PermissionSchema::default_schema();

        // Check document type exists
        let doc_type = schema.get_type("document");
        assert!(doc_type.is_some());

        let doc_type = doc_type.expect("document type should exist");

        // Check owner relation
        let owner_rel = doc_type.get_relation("owner");
        assert!(owner_rel.is_some());

        // Check viewer relation
        let viewer_rel = doc_type.get_relation("viewer");
        assert!(viewer_rel.is_some());

        let viewer_rel = viewer_rel.expect("viewer relation should exist");
        assert!(viewer_rel.rewrite.is_union());
    }

    #[test]
    fn test_custom_schema() {
        let mut schema = PermissionSchema::new();

        let mut my_type = TypeDefinition::new();
        my_type.add_relation(
            "admin",
            RelationDefinition {
                description: Some("Administrator".to_string()),
                rewrite: RewriteRule::This,
            },
        );

        schema.add_type("my_resource", my_type);

        let resource_type = schema.get_type("my_resource");
        assert!(resource_type.is_some());
    }

    #[test]
    fn test_rewrite_rule_checks() {
        let this_rule = RewriteRule::This;
        assert!(this_rule.is_this());

        let union_rule = RewriteRule::Union(vec![RewriteRule::This]);
        assert!(union_rule.is_union());

        let computed_rule = RewriteRule::ComputedUserset {
            relation: "owner".to_string(),
        };
        assert!(computed_rule.is_computed_userset());
    }

    #[test]
    fn test_default_schema_types() {
        let schema = PermissionSchema::default_schema();

        // Verify all default types exist
        assert!(schema.get_type("document").is_some());
        assert!(schema.get_type("folder").is_some());
        assert!(schema.get_type("team").is_some());
        assert!(schema.get_type("tenant").is_some());
    }
}
