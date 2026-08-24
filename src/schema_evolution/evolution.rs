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
            // A rename breaks anything (code, downstream schemas, saved
            // queries) that still refers to the old column name -- it is
            // not a purely additive change even though no data is lost.
            SchemaChange::RenameColumn { .. } => true,
            SchemaChange::ChangeType { .. } => true,
            SchemaChange::SetNullable { nullable, .. } => !nullable, // Making non-nullable is breaking
            SchemaChange::AddConstraint { .. } => true,
            // Adding a column is only safe for existing rows if a value can
            // be materialised for them: nullable columns get NA/a default,
            // but a *required* column with no default has nothing to put in
            // rows that predate the migration.
            SchemaChange::AddColumn { schema, .. } => {
                !schema.nullable && schema.default_value.is_none()
            }
            _ => false,
        }
    }

    /// Returns the inverse of this change, if one can be computed generically.
    ///
    /// Only changes that carry enough information to undo themselves have an
    /// inverse (e.g. a rename can be reversed because both names are known).
    /// Changes that discard information needed to reconstruct the prior state
    /// (e.g. removing a column: its original [`ColumnSchema`] is not part of
    /// `RemoveColumn`) return `Err` naming what's missing, rather than
    /// fabricating a plausible-looking but wrong inverse.
    pub fn inverse(&self) -> Result<SchemaChange, String> {
        match self {
            SchemaChange::AddColumn { schema, .. } => Ok(SchemaChange::RemoveColumn {
                name: schema.name.clone(),
            }),
            SchemaChange::RenameColumn { from, to } => Ok(SchemaChange::RenameColumn {
                from: to.clone(),
                to: from.clone(),
            }),
            SchemaChange::AddColumnTag { column, tag } => Ok(SchemaChange::RemoveColumnTag {
                column: column.clone(),
                tag: tag.clone(),
            }),
            SchemaChange::RemoveColumnTag { column, tag } => Ok(SchemaChange::AddColumnTag {
                column: column.clone(),
                tag: tag.clone(),
            }),
            SchemaChange::SetNullable { column, nullable } => Ok(SchemaChange::SetNullable {
                column: column.clone(),
                nullable: !nullable,
            }),
            SchemaChange::RemoveColumn { name } => Err(format!(
                "cannot invert RemoveColumn('{}'): the removed column's original \
                 definition (type, nullability, default) is not recorded on this change",
                name
            )),
            SchemaChange::ChangeType { column, .. } => Err(format!(
                "cannot invert ChangeType('{}'): the column's type before this change \
                 is not recorded on this change",
                column
            )),
            SchemaChange::RemoveConstraint { constraint_id } => Err(format!(
                "cannot invert RemoveConstraint('{}'): the removed constraint's \
                 definition is not recorded on this change, only its id",
                constraint_id
            )),
            SchemaChange::SetDefault { column, .. } => Err(format!(
                "cannot invert SetDefault('{}'): the column's previous default is not \
                 recorded on this change",
                column
            )),
            SchemaChange::SetColumnDescription { column, .. } => Err(format!(
                "cannot invert SetColumnDescription('{}'): the column's previous \
                 description is not recorded on this change",
                column
            )),
            SchemaChange::ReorderColumns { .. } => Err(
                "cannot invert ReorderColumns: the order before this change is not \
                 recorded on this change"
                    .to_string(),
            ),
            SchemaChange::AddConstraint { constraint } => Err(format!(
                "cannot invert AddConstraint({}): removing it would require its \
                 generated id, and doing so silently would hide that the constraint \
                 existed post-migration -- call this out explicitly if you rely on it",
                constraint
            )),
            SchemaChange::SetMetadata { key, .. } => Err(format!(
                "cannot invert SetMetadata('{}'): the previous value (or absence) of \
                 this key is not recorded on this change",
                key
            )),
            SchemaChange::RemoveMetadata { key } => Err(format!(
                "cannot invert RemoveMetadata('{}'): the removed value is not recorded \
                 on this change",
                key
            )),
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
    /// Name of the schema this migration applies to.
    ///
    /// This is what [`super::registry::SchemaRegistry::add_migration`] uses
    /// to index the migration for [`super::registry::SchemaRegistry::find_migration_path`],
    /// and what [`super::serialization::SchemaBundle`] uses to keep a
    /// migration associated with its schema across a save/load round trip.
    /// `#[serde(default)]` so migrations saved before this field existed
    /// still deserialize (as an empty string, which `validate()` rejects
    /// before the migration can be indexed -- it fails loudly rather than
    /// silently landing in the wrong schema's migration graph).
    #[serde(default)]
    pub schema_name: String,
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
    /// Create a new migration for the named schema
    pub fn new(
        id: impl Into<String>,
        schema_name: impl Into<String>,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
        description: impl Into<String>,
    ) -> Self {
        Migration {
            id: id.into(),
            schema_name: schema_name.into(),
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
        if self.schema_name.trim().is_empty() {
            return Err(format!(
                "Migration '{}' has no schema_name set; it cannot be indexed for \
                 path-finding (build it with a schema name via MigrationBuilder::new, \
                 or register it through SchemaRegistry::add_migration_for_schema)",
                self.id
            ));
        }
        if self.changes.is_empty() {
            return Err("Migration must have at least one change".to_string());
        }
        // `to` must be strictly newer than `from`: a migration that moves
        // backward or sideways can never appear on a forward path found by
        // SchemaRegistry::find_migration_path, so registering one is
        // always a mistake worth catching immediately.
        if self.to_version <= self.from_version {
            return Err(format!(
                "Migration '{}' target version {} must be greater than source version {}",
                self.id, self.to_version, self.from_version
            ));
        }

        // Internal consistency of the change list itself.
        let mut added_columns = std::collections::HashSet::new();
        let mut removed_columns = std::collections::HashSet::new();
        let mut retyped_columns = std::collections::HashSet::new();
        for change in &self.changes {
            match change {
                SchemaChange::RenameColumn { from, to } if from == to => {
                    return Err(format!(
                        "Migration '{}' renames column '{}' to itself",
                        self.id, from
                    ));
                }
                SchemaChange::AddColumn { schema, .. } => {
                    if !added_columns.insert(schema.name.clone()) {
                        return Err(format!(
                            "Migration '{}' adds column '{}' more than once",
                            self.id, schema.name
                        ));
                    }
                }
                SchemaChange::RemoveColumn { name } => {
                    if !removed_columns.insert(name.clone()) {
                        return Err(format!(
                            "Migration '{}' removes column '{}' more than once",
                            self.id, name
                        ));
                    }
                }
                SchemaChange::ChangeType { column, .. } => {
                    if !retyped_columns.insert(column.clone()) {
                        return Err(format!(
                            "Migration '{}' changes the type of column '{}' more than once",
                            self.id, column
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Compute the inverse of this migration: the changes that undo it,
    /// in reverse order, swapping `from_version`/`to_version`.
    ///
    /// Refuses (`Err`) when `reversible` is `false`, or when any individual
    /// change has no well-defined inverse (see [`SchemaChange::inverse`]) --
    /// the `reversible` flag is a promise about the *whole* migration, and a
    /// partial inverse would silently apply only some of the original
    /// changes' undo operations, leaving data in a state the caller did not
    /// ask for.
    pub fn inverse(&self) -> Result<Migration, String> {
        if !self.reversible {
            return Err(format!(
                "Migration '{}' is marked irreversible (reversible = false)",
                self.id
            ));
        }

        let mut inverted_changes = Vec::with_capacity(self.changes.len());
        for change in self.changes.iter().rev() {
            inverted_changes.push(change.inverse().map_err(|e| {
                format!(
                    "Migration '{}' cannot be inverted despite reversible = true: {}",
                    self.id, e
                )
            })?);
        }

        Ok(Migration {
            id: format!("{}_inverse", self.id),
            schema_name: self.schema_name.clone(),
            from_version: self.to_version.clone(),
            to_version: self.from_version.clone(),
            description: format!("Inverse of: {}", self.description),
            changes: inverted_changes,
            created_at: chrono::Utc::now().to_rfc3339(),
            author: self.author.clone(),
            reversible: true,
        })
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
    schema_name: String,
    from_version: SchemaVersion,
    to_version: SchemaVersion,
    description: String,
    changes: Vec<SchemaChange>,
    author: Option<String>,
    reversible: bool,
}

impl MigrationBuilder {
    /// Create a new migration builder for the named schema
    pub fn new(
        id: impl Into<String>,
        schema_name: impl Into<String>,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> Self {
        MigrationBuilder {
            id: id.into(),
            schema_name: schema_name.into(),
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
            schema_name: self.schema_name,
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
        // Renaming breaks anything (code, downstream schemas) still
        // addressing the column by its old name.
        assert!(rename.is_breaking());

        let required_no_default = SchemaChange::AddColumn {
            schema: ColumnSchema::new("required_col", SchemaDataType::String).with_nullable(false),
            position: None,
        };
        assert!(required_no_default.is_breaking());
    }

    #[test]
    fn test_schema_change_inverse() {
        let add = SchemaChange::AddColumn {
            schema: ColumnSchema::new("email", SchemaDataType::String),
            position: None,
        };
        assert_eq!(
            add.inverse(),
            Ok(SchemaChange::RemoveColumn {
                name: "email".to_string()
            })
        );

        let rename = SchemaChange::RenameColumn {
            from: "old".to_string(),
            to: "new".to_string(),
        };
        assert_eq!(
            rename.inverse(),
            Ok(SchemaChange::RenameColumn {
                from: "new".to_string(),
                to: "old".to_string(),
            })
        );

        // No original definition to restore -> not invertible.
        let remove = SchemaChange::RemoveColumn {
            name: "col".to_string(),
        };
        assert!(remove.inverse().is_err());
    }

    #[test]
    fn test_migration_builder() {
        let migration = MigrationBuilder::new(
            "m001",
            "users",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .description("Add email column")
        .author("admin")
        .add_column(ColumnSchema::new("email", SchemaDataType::String), None)
        .build();

        assert_eq!(migration.id, "m001");
        assert_eq!(migration.schema_name, "users");
        assert_eq!(migration.changes.len(), 1);
        assert!(!migration.has_breaking_changes());
    }

    #[test]
    fn test_migration_validate() {
        let valid = MigrationBuilder::new(
            "m001",
            "users",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .add_column(ColumnSchema::new("col", SchemaDataType::String), None)
        .build();
        assert!(valid.validate().is_ok());

        // No changes
        let invalid = Migration {
            id: "m002".to_string(),
            schema_name: "users".to_string(),
            from_version: SchemaVersion::new(1, 0, 0),
            to_version: SchemaVersion::new(1, 1, 0),
            description: "test".to_string(),
            changes: vec![],
            created_at: "2024-01-01T00:00:00Z".to_string(),
            author: None,
            reversible: true,
        };
        assert!(invalid.validate().is_err());

        // Missing schema_name (e.g. deserialized from a pre-field bundle)
        let no_schema_name = Migration {
            schema_name: String::new(),
            ..valid.clone()
        };
        assert!(no_schema_name.validate().is_err());

        // Backward (to <= from) is rejected
        let backward = Migration {
            from_version: SchemaVersion::new(2, 0, 0),
            to_version: SchemaVersion::new(1, 0, 0),
            ..valid.clone()
        };
        assert!(backward.validate().is_err());

        // Self-rename is rejected
        let self_rename = MigrationBuilder::new(
            "m003",
            "users",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .rename_column("a", "a")
        .build();
        assert!(self_rename.validate().is_err());
    }

    #[test]
    fn test_migration_inverse() {
        let migration = MigrationBuilder::new(
            "m001",
            "users",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .add_column(ColumnSchema::new("email", SchemaDataType::String), None)
        .rename_column("name", "full_name")
        .build();

        let inverse = migration.inverse().expect("reversible migration");
        assert_eq!(inverse.from_version, SchemaVersion::new(1, 1, 0));
        assert_eq!(inverse.to_version, SchemaVersion::new(1, 0, 0));
        assert_eq!(inverse.changes.len(), 2);
        // Applied in reverse order: undo the rename first, then the add.
        assert_eq!(
            inverse.changes[0],
            SchemaChange::RenameColumn {
                from: "full_name".to_string(),
                to: "name".to_string(),
            }
        );
        assert_eq!(
            inverse.changes[1],
            SchemaChange::RemoveColumn {
                name: "email".to_string()
            }
        );

        // Irreversible migrations refuse to invert.
        let irreversible = MigrationBuilder::new(
            "m002",
            "users",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .rename_column("a", "b")
        .irreversible()
        .build();
        assert!(irreversible.inverse().is_err());

        // A migration containing a non-invertible change refuses too, even
        // when marked reversible.
        let has_remove = MigrationBuilder::new(
            "m003",
            "users",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .remove_column("legacy")
        .build();
        assert!(has_remove.inverse().is_err());
    }
}
