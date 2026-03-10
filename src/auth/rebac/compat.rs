//! RBAC Compatibility Layer
//!
//! This module provides backward compatibility with traditional RBAC
//! by translating RBAC concepts (roles, permissions) into ReBAC relationships.

use super::manager::RebacManager;
use crate::error::{Error, Result};
use crate::multitenancy::Permission;
use std::sync::{Arc, RwLock};

/// RBAC to ReBAC compatibility layer
#[derive(Debug)]
pub struct RbacCompatLayer {
    /// Underlying ReBAC manager
    rebac: Arc<RwLock<RebacManager>>,
}

impl RbacCompatLayer {
    /// Create a new compatibility layer
    pub fn new(rebac: Arc<RwLock<RebacManager>>) -> Self {
        RbacCompatLayer { rebac }
    }

    /// Assign a role to a user for a tenant
    pub fn assign_role(&self, user_id: &str, role: &str, tenant_id: &str) -> Result<()> {
        let rebac = self
            .rebac
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        let user_subject = format!("user:{}", user_id);
        let tenant_object = format!("tenant:{}", tenant_id);

        // Map RBAC roles to ReBAC relationships
        match role.to_lowercase().as_str() {
            "admin" => {
                // Admin has all permissions (including inherited ones)
                rebac.grant(&user_subject, "admin", &tenant_object)?;
                rebac.grant(&user_subject, "owner", &tenant_object)?;
                rebac.grant(&user_subject, "manager", &tenant_object)?;
                rebac.grant(&user_subject, "editor", &tenant_object)?;
                rebac.grant(&user_subject, "viewer", &tenant_object)?;
            }
            "manager" => {
                // Manager can manage resources (includes editor and viewer)
                rebac.grant(&user_subject, "manager", &tenant_object)?;
                rebac.grant(&user_subject, "editor", &tenant_object)?;
                rebac.grant(&user_subject, "viewer", &tenant_object)?;
            }
            "analyst" | "editor" => {
                // Analyst/Editor can read and write (includes viewer)
                rebac.grant(&user_subject, "editor", &tenant_object)?;
                rebac.grant(&user_subject, "viewer", &tenant_object)?;
            }
            "viewer" => {
                // Viewer can only read
                rebac.grant(&user_subject, "viewer", &tenant_object)?;
            }
            _ => {
                return Err(Error::InvalidInput(format!("Unknown role: {}", role)));
            }
        }

        Ok(())
    }

    /// Remove a role from a user for a tenant
    pub fn revoke_role(&self, user_id: &str, role: &str, tenant_id: &str) -> Result<()> {
        let rebac = self
            .rebac
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        let user_subject = format!("user:{}", user_id);
        let tenant_object = format!("tenant:{}", tenant_id);

        match role.to_lowercase().as_str() {
            "admin" => {
                let _ = rebac.revoke(&user_subject, "admin", &tenant_object);
                let _ = rebac.revoke(&user_subject, "owner", &tenant_object);
                let _ = rebac.revoke(&user_subject, "manager", &tenant_object);
                let _ = rebac.revoke(&user_subject, "editor", &tenant_object);
                let _ = rebac.revoke(&user_subject, "viewer", &tenant_object);
            }
            "manager" => {
                let _ = rebac.revoke(&user_subject, "manager", &tenant_object);
                let _ = rebac.revoke(&user_subject, "editor", &tenant_object);
                let _ = rebac.revoke(&user_subject, "viewer", &tenant_object);
            }
            "analyst" | "editor" => {
                let _ = rebac.revoke(&user_subject, "editor", &tenant_object);
                let _ = rebac.revoke(&user_subject, "viewer", &tenant_object);
            }
            "viewer" => {
                let _ = rebac.revoke(&user_subject, "viewer", &tenant_object);
            }
            _ => {
                return Err(Error::InvalidInput(format!("Unknown role: {}", role)));
            }
        }

        Ok(())
    }

    /// Check if user has a permission on a resource in a tenant
    pub fn check_permission(
        &self,
        user_id: &str,
        permission: &Permission,
        resource_type: &str,
        resource_id: &str,
        tenant_id: &str,
    ) -> Result<bool> {
        let rebac = self
            .rebac
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        let user_subject = format!("user:{}", user_id);
        let resource_object = format!("{}:{}", resource_type, resource_id);

        // Map permission to relation
        let relation = self.map_permission_to_relation(permission);

        // Check direct permission on resource
        let has_direct = rebac.check_access(&user_subject, relation, &resource_object)?;
        if has_direct {
            return Ok(true);
        }

        // Check tenant-level permission
        let tenant_object = format!("tenant:{}", tenant_id);
        let has_tenant = rebac.check_access(&user_subject, relation, &tenant_object)?;

        Ok(has_tenant)
    }

    /// Check if user has any of the given permissions
    pub fn check_any_permission(
        &self,
        user_id: &str,
        permissions: &[Permission],
        resource_type: &str,
        resource_id: &str,
        tenant_id: &str,
    ) -> Result<bool> {
        for permission in permissions {
            if self.check_permission(user_id, permission, resource_type, resource_id, tenant_id)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Check if user has all of the given permissions
    pub fn check_all_permissions(
        &self,
        user_id: &str,
        permissions: &[Permission],
        resource_type: &str,
        resource_id: &str,
        tenant_id: &str,
    ) -> Result<bool> {
        for permission in permissions {
            if !self.check_permission(user_id, permission, resource_type, resource_id, tenant_id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// List all resources of a type that user has permission on
    pub fn list_accessible_resources(
        &self,
        user_id: &str,
        permission: &Permission,
        resource_type: &str,
    ) -> Result<Vec<String>> {
        let rebac = self
            .rebac
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        let user_subject = format!("user:{}", user_id);
        let relation = self.map_permission_to_relation(permission);

        rebac.list_accessible(&user_subject, relation, resource_type)
    }

    /// Get all users with a specific role in a tenant
    pub fn get_users_with_role(&self, role: &str, tenant_id: &str) -> Result<Vec<String>> {
        let rebac = self
            .rebac
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        let tenant_object = format!("tenant:{}", tenant_id);

        // Map role to primary relation
        let relation = match role.to_lowercase().as_str() {
            "admin" => "admin",
            "manager" => "manager",
            "analyst" | "editor" => "editor",
            "viewer" => "viewer",
            _ => return Err(Error::InvalidInput(format!("Unknown role: {}", role))),
        };

        let subjects = rebac.expand(relation, &tenant_object)?;

        // Extract user IDs from subjects
        let user_ids: Vec<String> = subjects
            .into_iter()
            .filter_map(|s| {
                if s.starts_with("user:") {
                    Some(s.strip_prefix("user:").unwrap_or(&s).to_string())
                } else {
                    None
                }
            })
            .collect();

        Ok(user_ids)
    }

    /// Map Permission enum to relation string
    fn map_permission_to_relation(&self, permission: &Permission) -> &str {
        match permission {
            Permission::Read => "viewer",
            Permission::Write => "editor",
            Permission::Delete => "owner",
            Permission::Create => "editor",
            Permission::Share => "owner",
            Permission::Admin => "admin",
        }
    }

    /// Create resource-level permissions
    pub fn grant_resource_permission(
        &self,
        user_id: &str,
        permission: &Permission,
        resource_type: &str,
        resource_id: &str,
    ) -> Result<()> {
        let rebac = self
            .rebac
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        let user_subject = format!("user:{}", user_id);
        let resource_object = format!("{}:{}", resource_type, resource_id);
        let relation = self.map_permission_to_relation(permission);

        rebac.grant(&user_subject, relation, &resource_object)
    }

    /// Revoke resource-level permissions
    pub fn revoke_resource_permission(
        &self,
        user_id: &str,
        permission: &Permission,
        resource_type: &str,
        resource_id: &str,
    ) -> Result<()> {
        let rebac = self
            .rebac
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        let user_subject = format!("user:{}", user_id);
        let resource_object = format!("{}:{}", resource_type, resource_id);
        let relation = self.map_permission_to_relation(permission);

        rebac.revoke(&user_subject, relation, &resource_object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_compat() -> RbacCompatLayer {
        let rebac = Arc::new(RwLock::new(RebacManager::new()));
        RbacCompatLayer::new(rebac)
    }

    #[test]
    fn test_assign_role() {
        let compat = create_test_compat();

        compat
            .assign_role("alice", "admin", "tenant_a")
            .expect("assign should succeed");

        // Admin should have admin permission
        let has_admin = compat
            .check_permission(
                "alice",
                &Permission::Admin,
                "tenant",
                "tenant_a",
                "tenant_a",
            )
            .expect("check should succeed");
        assert!(has_admin);
    }

    #[test]
    fn test_revoke_role() {
        let compat = create_test_compat();

        compat
            .assign_role("alice", "admin", "tenant_a")
            .expect("assign should succeed");

        compat
            .revoke_role("alice", "admin", "tenant_a")
            .expect("revoke should succeed");

        let has_admin = compat
            .check_permission(
                "alice",
                &Permission::Admin,
                "tenant",
                "tenant_a",
                "tenant_a",
            )
            .expect("check should succeed");
        assert!(!has_admin);
    }

    #[test]
    fn test_check_permission() {
        let compat = create_test_compat();

        compat
            .assign_role("alice", "viewer", "tenant_a")
            .expect("assign should succeed");

        let can_read = compat
            .check_permission("alice", &Permission::Read, "document", "123", "tenant_a")
            .expect("check should succeed");
        assert!(can_read);

        let can_write = compat
            .check_permission("alice", &Permission::Write, "document", "123", "tenant_a")
            .expect("check should succeed");
        assert!(!can_write);
    }

    #[test]
    fn test_resource_level_permission() {
        let compat = create_test_compat();

        compat
            .grant_resource_permission("alice", &Permission::Write, "document", "123")
            .expect("grant should succeed");

        let rebac = compat.rebac.read().expect("lock should succeed");
        let has_access = rebac
            .check_access("user:alice", "editor", "document:123")
            .expect("check should succeed");
        assert!(has_access);
    }

    #[test]
    fn test_check_any_permission() {
        let compat = create_test_compat();

        compat
            .assign_role("alice", "editor", "tenant_a")
            .expect("assign should succeed");

        let has_any = compat
            .check_any_permission(
                "alice",
                &[Permission::Write, Permission::Delete],
                "document",
                "123",
                "tenant_a",
            )
            .expect("check should succeed");
        assert!(has_any);
    }

    #[test]
    fn test_check_all_permissions() {
        let compat = create_test_compat();

        compat
            .assign_role("alice", "admin", "tenant_a")
            .expect("assign should succeed");

        let has_all = compat
            .check_all_permissions(
                "alice",
                &[Permission::Read, Permission::Write],
                "document",
                "123",
                "tenant_a",
            )
            .expect("check should succeed");
        assert!(has_all);
    }

    #[test]
    fn test_get_users_with_role() {
        let compat = create_test_compat();

        compat
            .assign_role("alice", "admin", "tenant_a")
            .expect("assign should succeed");
        compat
            .assign_role("bob", "admin", "tenant_a")
            .expect("assign should succeed");

        let admins = compat
            .get_users_with_role("admin", "tenant_a")
            .expect("get should succeed");

        assert_eq!(admins.len(), 2);
        assert!(admins.contains(&"alice".to_string()));
        assert!(admins.contains(&"bob".to_string()));
    }

    #[test]
    fn test_role_hierarchy() {
        let compat = create_test_compat();

        // Assign manager role
        compat
            .assign_role("alice", "manager", "tenant_a")
            .expect("assign should succeed");

        // Manager should have write permission
        let can_write = compat
            .check_permission("alice", &Permission::Write, "document", "123", "tenant_a")
            .expect("check should succeed");
        assert!(can_write);

        // But not admin permission
        let is_admin = compat
            .check_permission("alice", &Permission::Admin, "document", "123", "tenant_a")
            .expect("check should succeed");
        assert!(!is_admin);
    }
}
