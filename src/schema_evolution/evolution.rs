//! Schema evolution types - changes and migrations
//!
//! This module defines the types that represent schema changes (individual
//! mutations to a schema) and migrations (ordered collections of changes
//! that move a schema from one version to another).

use serde::{Deserialize, Serialize};
use std::fmt;

use super::schema::{ColumnSchema, DefaultValue, SchemaConstraint, SchemaDataType, SchemaVersion};

/// A single change operation that can be applied to a schema
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaChange {
    /// Add a new column, optionally at a specific position
    AddColumn {
        /// Column definition to add
        schema: ColumnSchema,
        /// Optional position (None = append at end)
        position: Option<usize>,
    },
    /// Remove an existing column
    RemoveColumn {
        /// Name of column to remove
        name: String,
    },
    /// Rename an existing column
    RenameColumn {
        /// Current column name
        from: String,
        /// New column name
        to: String,
    },
    /// Change the data type of a column
    ChangeType {
        /// Column name
        column: String,
        /// New data type
        new_type: SchemaDataType,
        /// Optional converter identifier (used by the migrator to look up a conversion function)
        converter: Option<String>,
    },
    /// Add a constraint to the schema
    AddConstraint {
        /// The constraint to add
        constraint: SchemaConstraint,
    },
    /// Remove a constraint by its generated ID
    RemoveConstraint {
        /// Constraint ID (as returned by SchemaConstraint::generate_id)
        constraint_id: String,
    },
    /// Set or update the default value for a column
    SetDefault {
        /// Column name
        column: String,
        /// New default value
        default: DefaultValue,
    },
    /// Change the nullability of a column
    SetNullable {
        /// Column name
        column: String,
        /// New nullability setting
        nullable: bool,
    },
    /// Set the description of a column
    SetColumnDescription {
        /// Column name
        column: String,
        /// New description
        description: String,
    },
    /// Reorder columns to match the given order
    ReorderColumns {
        /// Desired column order (all columns must be listed)
        order: Vec<String>,
    },
    /// Add a tag to a column
    AddColumnTag {
        /// Column name
        column: String,
        /// Tag to add
        tag: String,
    },
    /// Remove a tag from a column
    RemoveColumnTag {
        /// Column name
        column: String,
        /// Tag to remove
        tag: String,
    },
    /// Update schema metadata
    SetMetadata {
        /// Metadata key
        key: String,
        /// Metadata value
        value: String,
    },
    /// Remove schema metadata
    RemoveMetadata {
        /// Metadata key to remove
        key: String,
    },
}

impl SchemaChange {
    /// Returns a human-readable description of this change
    pub fn describe(&self) -> String {
        match self {
            SchemaChange::AddColumn { schema, position } => {
                if let Some(pos) = position {
                    format!(
                        "Add column '{}' ({}) at position {}",
                        schema.name, schema.data_type, pos
                    )
                } else {
                    format!("Add column '{}' ({})", schema.name, schema.data_type)
                }
            }
            SchemaChange::RemoveColumn { name } => {
                format!("Remove column '{}'", name)
            }
            SchemaChange::RenameColumn { from, to } => {
                format!("Rename column '{}' to '{}'", from, to)
            }
            SchemaChange::ChangeType {
                column, new_type, ..
            } => {
                format!("Change type of column '{}' to {}", column, new_type)
            }
            SchemaChange::AddConstraint { constraint } => {
                format!("Add constraint: {}", constraint)
            }
            SchemaChange::RemoveConstraint { constraint_id } => {
                format!("Remove constraint '{}'", constraint_id)
            }
            SchemaChange::SetDefault { column, default } => {
                format!("Set default for '{}' to {}", column, default)
            }
            SchemaChange::SetNullable { column, nullable } => {
                if *nullable {
                    format!("Make column '{}' nullable", column)
                } else {
                    format!("Make column '{}' non-nullable", column)
                }
            }
            SchemaChange::SetColumnDescription {
                column,
                description,
            } => {
                format!("Set description of '{}' to: {}", column, description)
            }
            SchemaChange::ReorderColumns { order } => {
                format!("Reorder columns: {}", order.join(", "))
            }
            SchemaChange::AddColumnTag { column, tag } => {
                format!("Add tag '{}' to column '{}'", tag, column)
            }
            SchemaChange::RemoveColumnTag { column, tag } => {
                format!("Remove tag '{}' from column '{}'", tag, column)
            }
            SchemaChange::SetMetadata { key, value } => {
                format!("Set metadata '{}' = '{}'", key, value)
            }
            SchemaChange::RemoveMetadata { key } => {
                format!("Remove metadata key '{}'", key)
            }
        }
    }

    /// Returns whether this change is potentially breaking (may cause data loss or incompatibility)
    pub fn is_breaking(&self) -> bool {
        match self {
            SchemaChange::RemoveColumn { .. } => true,
            SchemaChange::ChangeType { .. } => true,
            SchemaChange::SetNullable { nullable, .. } => !nullable, // Making non-nullable is breaking
            SchemaChange::AddConstraint { .. } => true,
            _ => false,
        }
    }

    /// Returns the columns affected by this change
    pub fn affected_columns(&self) -> Vec<&str> {
        match self {
            SchemaChange::AddColumn { schema, .. } => vec![schema.name.as_str()],
            SchemaChange::RemoveColumn { name } => vec![name.as_str()],
            SchemaChange::RenameColumn { from, .. } => vec![from.as_str()],
            SchemaChange::ChangeType { column, .. } => vec![column.as_str()],
            SchemaChange::AddConstraint { constraint } => constraint.affected_columns(),
            SchemaChange::RemoveConstraint { .. } => vec![],
            SchemaChange::SetDefault { column, .. } => vec![column.as_str()],
            SchemaChange::SetNullable { column, .. } => vec![column.as_str()],
            SchemaChange::SetColumnDescription { column, .. } => vec![column.as_str()],
            SchemaChange::ReorderColumns { order } => order.iter().map(|s| s.as_str()).collect(),
            SchemaChange::AddColumnTag { column, .. } => vec![column.as_str()],
            SchemaChange::RemoveColumnTag { column, .. } => vec![column.as_str()],
            SchemaChange::SetMetadata { .. } => vec![],
            SchemaChange::RemoveMetadata { .. } => vec![],
        }
    }
}

impl fmt::Display for SchemaChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// A complete migration from one schema version to another
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Migration {
    /// Unique identifier for this migration
    pub id: String,
    /// Source schema version
    pub from_version: SchemaVersion,
    /// Target schema version
    pub to_version: SchemaVersion,
    /// Human-readable description of what this migration does
    pub description: String,
    /// The ordered list of changes to apply
    pub changes: Vec<SchemaChange>,
    /// ISO 8601 timestamp when this migration was created
    pub created_at: String,
    /// Optional author of this migration
    pub author: Option<String>,
    /// Whether this migration is reversible
    pub reversible: bool,
}

impl Migration {
    /// Create a new migration
    pub fn new(
        id: impl Into<String>,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
        description: impl Into<String>,
    ) -> Self {
        Migration {
            id: id.into(),
            from_version,
            to_version,
            description: description.into(),
            changes: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            author: None,
            reversible: true,
        }
    }

    /// Add a change to this migration
    pub fn with_change(mut self, change: SchemaChange) -> Self {
        self.changes.push(change);
        self
    }

    /// Add multiple changes
    pub fn with_changes(mut self, changes: Vec<SchemaChange>) -> Self {
        self.changes.extend(changes);
        self
    }

    /// Set the author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Mark as irreversible
    pub fn irreversible(mut self) -> Self {
        self.reversible = false;
        self
    }

    /// Check if this migration has any breaking changes
    pub fn has_breaking_changes(&self) -> bool {
        self.changes.iter().any(|c| c.is_breaking())
    }

    /// Get all breaking changes
    pub fn breaking_changes(&self) -> Vec<&SchemaChange> {
        self.changes.iter().filter(|c| c.is_breaking()).collect()
    }

    /// Validate the migration's internal consistency
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("Migration ID cannot be empty".to_string());
        }
        if self.changes.is_empty() {
            return Err("Migration must have at least one change".to_string());
        }
        if self.from_version == self.to_version {
            return Err("From and to versions must differ".to_string());
        }
        Ok(())
    }
}

impl fmt::Display for Migration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Migration {} (v{} -> v{}): {}",
            self.id, self.from_version, self.to_version, self.description
        )?;
        writeln!(f, "Changes ({}):", self.changes.len())?;
        for change in &self.changes {
            writeln!(f, "  - {}", change)?;
        }
        Ok(())
    }
}

/// Builder for creating migrations fluently
pub struct MigrationBuilder {
    id: String,
    from_version: SchemaVersion,
    to_version: SchemaVersion,
    description: String,
    changes: Vec<SchemaChange>,
    author: Option<String>,
    reversible: bool,
}

impl MigrationBuilder {
    /// Create a new migration builder
    pub fn new(
        id: impl Into<String>,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> Self {
        MigrationBuilder {
            id: id.into(),
            from_version,
            to_version,
            description: String::new(),
            changes: Vec::new(),
            author: None,
            reversible: true,
        }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Mark as irreversible
    pub fn irreversible(mut self) -> Self {
        self.reversible = false;
        self
    }

    /// Add a column
    pub fn add_column(mut self, schema: ColumnSchema, position: Option<usize>) -> Self {
        self.changes
            .push(SchemaChange::AddColumn { schema, position });
        self
    }

    /// Remove a column
    pub fn remove_column(mut self, name: impl Into<String>) -> Self {
        self.changes
            .push(SchemaChange::RemoveColumn { name: name.into() });
        self
    }

    /// Rename a column
    pub fn rename_column(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.changes.push(SchemaChange::RenameColumn {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// Change a column's type
    pub fn change_type(
        mut self,
        column: impl Into<String>,
        new_type: SchemaDataType,
        converter: Option<String>,
    ) -> Self {
        self.changes.push(SchemaChange::ChangeType {
            column: column.into(),
            new_type,
            converter,
        });
        self
    }

    /// Add a constraint
    pub fn add_constraint(mut self, constraint: SchemaConstraint) -> Self {
        self.changes
            .push(SchemaChange::AddConstraint { constraint });
        self
    }

    /// Remove a constraint
    pub fn remove_constraint(mut self, constraint_id: impl Into<String>) -> Self {
        self.changes.push(SchemaChange::RemoveConstraint {
            constraint_id: constraint_id.into(),
        });
        self
    }

    /// Set default value
    pub fn set_default(mut self, column: impl Into<String>, default: DefaultValue) -> Self {
        self.changes.push(SchemaChange::SetDefault {
            column: column.into(),
            default,
        });
        self
    }

    /// Set nullability
    pub fn set_nullable(mut self, column: impl Into<String>, nullable: bool) -> Self {
        self.changes.push(SchemaChange::SetNullable {
            column: column.into(),
            nullable,
        });
        self
    }

    /// Build the migration
    pub fn build(self) -> Migration {
        Migration {
            id: self.id,
            from_version: self.from_version,
            to_version: self.to_version,
            description: self.description,
            changes: self.changes,
            created_at: chrono::Utc::now().to_rfc3339(),
            author: self.author,
            reversible: self.reversible,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_evolution::schema::{ColumnSchema, SchemaDataType, SchemaVersion};

    #[test]
    fn test_schema_change_is_breaking() {
        let remove = SchemaChange::RemoveColumn {
            name: "col".to_string(),
        };
        assert!(remove.is_breaking());

        let add = SchemaChange::AddColumn {
            schema: ColumnSchema::new("new_col", SchemaDataType::String),
            position: None,
        };
        assert!(!add.is_breaking());

        let rename = SchemaChange::RenameColumn {
            from: "old".to_string(),
            to: "new".to_string(),
        };
        assert!(!rename.is_breaking());
    }

    #[test]
    fn test_migration_builder() {
        let migration = MigrationBuilder::new(
            "m001",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .description("Add email column")
        .author("admin")
        .add_column(ColumnSchema::new("email", SchemaDataType::String), None)
        .build();

        assert_eq!(migration.id, "m001");
        assert_eq!(migration.changes.len(), 1);
        assert!(!migration.has_breaking_changes());
    }

    #[test]
    fn test_migration_validate() {
        let valid = MigrationBuilder::new(
            "m001",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .add_column(ColumnSchema::new("col", SchemaDataType::String), None)
        .build();
        assert!(valid.validate().is_ok());

        // No changes
        let invalid = Migration {
            id: "m002".to_string(),
            from_version: SchemaVersion::new(1, 0, 0),
            to_version: SchemaVersion::new(1, 1, 0),
            description: "test".to_string(),
            changes: vec![],
            created_at: "2024-01-01T00:00:00Z".to_string(),
            author: None,
            reversible: true,
        };
        assert!(invalid.validate().is_err());
    }
}
