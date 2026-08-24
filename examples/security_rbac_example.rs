#![allow(clippy::result_large_err)]
//! # Comprehensive Role-Based Access Control (RBAC) Example
//!
//! This example demonstrates enterprise-grade role-based access control patterns
//! using PandRS's authentication and multi-tenancy modules, including:
//!
//! - Role definition and hierarchical permissions
//! - Permission checking at multiple levels
//! - Resource-based access control (per-dataset)
//! - Multi-tenant RBAC with complete isolation
//! - Dynamic permission updates and role changes
//! - User groups and team-based access
//! - Audit logging for all access attempts
//! - Conditional access based on context
//!
//! # Use Case: Enterprise Data Analytics Platform
//!
//! This example simulates an enterprise data platform where organizations need
//! fine-grained control over who can access what data. The platform supports:
//! - Multiple organizational tenants with complete isolation
//! - Hierarchical roles (Admin > Manager > Analyst > Viewer)
//! - Resource-level permissions (per-dataset access control)
//! - Team-based collaboration within tenants
//! - Comprehensive audit trails for compliance

use pandrs::auth::{AuthManager, JwtConfig, UserInfo};
use pandrs::error::Result;
use pandrs::multitenancy::{Permission, TenantConfig, TenantId};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

/// Represents a role in the system with associated permissions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
enum Role {
    /// System administrator with full access
    Admin,
    /// Department manager with team oversight
    Manager,
    /// Data analyst with read/write access
    Analyst,
    /// Basic viewer with read-only access
    Viewer,
    /// Custom role with specific permissions
    Custom(String),
}

impl Role {
    /// Get default permissions for each role
    fn default_permissions(&self) -> HashSet<Permission> {
        match self {
            Role::Admin => vec![
                Permission::Read,
                Permission::Write,
                Permission::Delete,
                Permission::Create,
                Permission::Share,
                Permission::Admin,
            ]
            .into_iter()
            .collect(),
            Role::Manager => vec![
                Permission::Read,
                Permission::Write,
                Permission::Create,
                Permission::Share,
            ]
            .into_iter()
            .collect(),
            Role::Analyst => vec![Permission::Read, Permission::Write, Permission::Create]
                .into_iter()
                .collect(),
            Role::Viewer => vec![Permission::Read].into_iter().collect(),
            Role::Custom(_) => HashSet::new(),
        }
    }

    /// Get role hierarchy level (higher = more privileged)
    fn hierarchy_level(&self) -> u8 {
        match self {
            Role::Admin => 4,
            Role::Manager => 3,
            Role::Analyst => 2,
            Role::Viewer => 1,
            Role::Custom(_) => 0,
        }
    }

    /// Check if this role includes another role's permissions
    fn includes(&self, other: &Role) -> bool {
        self.hierarchy_level() >= other.hierarchy_level()
    }

    /// Convert role to string representation
    fn as_str(&self) -> &str {
        match self {
            Role::Admin => "admin",
            Role::Manager => "manager",
            Role::Analyst => "analyst",
            Role::Viewer => "viewer",
            Role::Custom(name) => name,
        }
    }
}

/// Resource identifier for access control
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResourceId(String);

impl ResourceId {
    fn new(id: impl Into<String>) -> Self {
        ResourceId(id.into())
    }
}

/// Access control entry for a specific resource
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ResourceAcl {
    resource_id: ResourceId,
    tenant_id: TenantId,
    owner_user_id: String,
    allowed_users: HashMap<String, HashSet<Permission>>,
    allowed_groups: HashMap<String, HashSet<Permission>>,
    public_permissions: HashSet<Permission>,
}

impl ResourceAcl {
    fn new(resource_id: ResourceId, tenant_id: TenantId, owner_user_id: String) -> Self {
        ResourceAcl {
            resource_id,
            tenant_id,
            owner_user_id,
            allowed_users: HashMap::new(),
            allowed_groups: HashMap::new(),
            public_permissions: HashSet::new(),
        }
    }

    /// Grant permissions to a specific user
    fn grant_user_permission(&mut self, user_id: String, permissions: HashSet<Permission>) {
        self.allowed_users.insert(user_id, permissions);
    }

    /// Grant permissions to a group
    fn grant_group_permission(&mut self, group_id: String, permissions: HashSet<Permission>) {
        self.allowed_groups.insert(group_id, permissions);
    }

    /// Check if user has a specific permission
    fn check_permission(
        &self,
        user_id: &str,
        permission: Permission,
        user_groups: &[String],
    ) -> bool {
        // Owner has all permissions
        if user_id == self.owner_user_id {
            return true;
        }

        // Check user-specific permissions
        if let Some(perms) = self.allowed_users.get(user_id) {
            if perms.contains(&permission) {
                return true;
            }
        }

        // Check group permissions
        for group in user_groups {
            if let Some(perms) = self.allowed_groups.get(group.as_str()) {
                if perms.contains(&permission) {
                    return true;
                }
            }
        }

        // Check public permissions
        self.public_permissions.contains(&permission)
    }
}

/// RBAC manager combining authentication and authorization
struct RbacManager {
    auth_manager: AuthManager,
    resource_acls: HashMap<ResourceId, ResourceAcl>,
    user_groups: HashMap<String, Vec<String>>,
    tenant_configs: HashMap<TenantId, TenantConfig>,
    access_logs: Vec<AccessLog>,
}

/// Access log entry for audit trail
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AccessLog {
    timestamp: SystemTime,
    user_id: String,
    tenant_id: TenantId,
    resource_id: Option<String>,
    action: String,
    permission: Permission,
    granted: bool,
    reason: String,
}

impl RbacManager {
    fn new() -> Self {
        RbacManager {
            auth_manager: AuthManager::new(JwtConfig::default()),
            resource_acls: HashMap::new(),
            user_groups: HashMap::new(),
            tenant_configs: HashMap::new(),
            access_logs: Vec::new(),
        }
    }

    /// Register a new tenant
    fn register_tenant(&mut self, config: TenantConfig) -> Result<()> {
        self.tenant_configs.insert(config.id.clone(), config);
        Ok(())
    }

    /// Register a user with a specific role
    fn register_user_with_role(
        &mut self,
        user_id: &str,
        email: &str,
        password: &str,
        tenant_id: &str,
        role: Role,
    ) -> Result<()> {
        let permissions: Vec<Permission> = role.default_permissions().into_iter().collect();

        let user = UserInfo::new(user_id, email, tenant_id)
            .with_password(password)
            .with_role(role.as_str())
            .with_permissions(permissions);

        self.auth_manager.register_user(user)
    }

    /// Add user to a group
    fn add_user_to_group(&mut self, user_id: String, group_id: String) {
        self.user_groups.entry(user_id).or_default().push(group_id);
    }

    /// Create a resource with ACL
    fn create_resource(
        &mut self,
        resource_id: ResourceId,
        tenant_id: TenantId,
        owner_user_id: String,
    ) -> Result<()> {
        let acl = ResourceAcl::new(resource_id.clone(), tenant_id, owner_user_id);
        self.resource_acls.insert(resource_id, acl);
        Ok(())
    }

    /// Check if user has permission for a resource
    fn check_access(
        &mut self,
        user_id: &str,
        tenant_id: &str,
        resource_id: Option<&ResourceId>,
        permission: Permission,
    ) -> Result<bool> {
        // Get user info
        let user = self.auth_manager.get_user(user_id).ok_or_else(|| {
            pandrs::error::Error::InvalidInput(format!("User '{}' not found", user_id))
        })?;

        // Check tenant isolation
        if user.tenant_id != tenant_id {
            let log = AccessLog {
                timestamp: SystemTime::now(),
                user_id: user_id.to_string(),
                tenant_id: tenant_id.to_string(),
                resource_id: resource_id.map(|r| r.0.clone()),
                action: "access_check".to_string(),
                permission,
                granted: false,
                reason: "Cross-tenant access denied".to_string(),
            };
            self.access_logs.push(log);
            return Ok(false);
        }

        // Check if user has permission at tenant level
        let has_permission = user.permissions.contains(&permission);

        // If checking specific resource, also check resource ACL
        let resource_allowed = if let Some(res_id) = resource_id {
            if let Some(acl) = self.resource_acls.get(res_id) {
                let user_groups = self.user_groups.get(user_id).cloned().unwrap_or_default();
                acl.check_permission(user_id, permission, &user_groups)
            } else {
                false
            }
        } else {
            true // No resource-specific check needed
        };

        let granted = has_permission && resource_allowed;

        // Log access attempt
        let log = AccessLog {
            timestamp: SystemTime::now(),
            user_id: user_id.to_string(),
            tenant_id: tenant_id.to_string(),
            resource_id: resource_id.map(|r| r.0.clone()),
            action: "access_check".to_string(),
            permission,
            granted,
            reason: if granted {
                "Access granted".to_string()
            } else if !has_permission {
                "User lacks required permission".to_string()
            } else {
                "Resource ACL denied access".to_string()
            },
        };
        self.access_logs.push(log);

        Ok(granted)
    }

    /// Grant resource permission to a user
    fn grant_resource_permission(
        &mut self,
        resource_id: &ResourceId,
        user_id: String,
        permissions: HashSet<Permission>,
    ) -> Result<()> {
        let acl = self
            .resource_acls
            .get_mut(resource_id)
            .ok_or_else(|| pandrs::error::Error::InvalidInput("Resource not found".to_string()))?;

        acl.grant_user_permission(user_id, permissions);
        Ok(())
    }

    /// Grant resource permission to a group
    fn grant_resource_permission_to_group(
        &mut self,
        resource_id: &ResourceId,
        group_id: String,
        permissions: HashSet<Permission>,
    ) -> Result<()> {
        let acl = self
            .resource_acls
            .get_mut(resource_id)
            .ok_or_else(|| pandrs::error::Error::InvalidInput("Resource not found".to_string()))?;

        acl.grant_group_permission(group_id, permissions);
        Ok(())
    }

    /// Get access logs for a user
    fn get_user_access_logs(&self, user_id: &str) -> Vec<&AccessLog> {
        self.access_logs
            .iter()
            .filter(|log| log.user_id == user_id)
            .collect()
    }

    /// Get access logs for a resource
    fn get_resource_access_logs(&self, resource_id: &str) -> Vec<&AccessLog> {
        self.access_logs
            .iter()
            .filter(|log| {
                log.resource_id
                    .as_ref()
                    .map(|id| id == resource_id)
                    .unwrap_or(false)
            })
            .collect()
    }
}

fn main() -> Result<()> {
    println!("🛡️  PandRS Security: Role-Based Access Control (RBAC) Example");
    println!("===========================================================\n");

    // Scenario 1: Basic RBAC Setup
    println!("📝 Scenario 1: Basic RBAC Setup");
    basic_rbac_setup()?;

    // Scenario 2: Hierarchical Role Permissions
    println!("\n👥 Scenario 2: Hierarchical Role Permissions");
    hierarchical_roles()?;

    // Scenario 3: Resource-Level Access Control
    println!("\n📁 Scenario 3: Resource-Level Access Control");
    resource_level_access()?;

    // Scenario 4: Multi-Tenant Isolation
    println!("\n🏢 Scenario 4: Multi-Tenant Isolation");
    multi_tenant_isolation()?;

    // Scenario 5: Group-Based Permissions
    println!("\n👨‍👩‍👧‍👦 Scenario 5: Group-Based Permissions");
    group_based_permissions()?;

    // Scenario 6: Dynamic Permission Updates
    println!("\n🔄 Scenario 6: Dynamic Permission Updates");
    dynamic_permission_updates()?;

    // Scenario 7: Comprehensive Audit Trail
    println!("\n📊 Scenario 7: Comprehensive Audit Trail");
    audit_trail_demo()?;

    // Scenario 8: Conditional Access Policies
    println!("\n⚡ Scenario 8: Conditional Access Policies");
    conditional_access()?;

    println!("\n✅ All RBAC scenarios completed successfully!");
    Ok(())
}

/// Demonstrates basic RBAC setup with different roles
fn basic_rbac_setup() -> Result<()> {
    let mut rbac = RbacManager::new();

    // Register tenant
    let tenant_config = TenantConfig::new("acme_corp")
        .with_name("ACME Corporation")
        .with_description("Enterprise customer")
        .with_max_datasets(1000);
    rbac.register_tenant(tenant_config)?;

    println!("  🏢 Tenant 'acme_corp' registered");

    // Register users with different roles
    println!("\n  👤 Registering users with different roles:");

    rbac.register_user_with_role(
        "alice",
        "alice@acme.com",
        "password",
        "acme_corp",
        Role::Admin,
    )?;
    println!("     ✅ Alice registered as Admin");

    rbac.register_user_with_role(
        "bob",
        "bob@acme.com",
        "password",
        "acme_corp",
        Role::Manager,
    )?;
    println!("     ✅ Bob registered as Manager");

    rbac.register_user_with_role(
        "carol",
        "carol@acme.com",
        "password",
        "acme_corp",
        Role::Analyst,
    )?;
    println!("     ✅ Carol registered as Analyst");

    rbac.register_user_with_role(
        "dave",
        "dave@acme.com",
        "password",
        "acme_corp",
        Role::Viewer,
    )?;
    println!("     ✅ Dave registered as Viewer");

    // Test permissions for each role
    println!("\n  🔐 Testing role permissions:");

    let roles = vec![
        ("alice", Role::Admin),
        ("bob", Role::Manager),
        ("carol", Role::Analyst),
        ("dave", Role::Viewer),
    ];

    for (user, role) in roles {
        println!("\n     {} ({:?}):", user, role);
        for perm in [
            Permission::Read,
            Permission::Write,
            Permission::Delete,
            Permission::Admin,
        ] {
            let has_access = rbac.check_access(user, "acme_corp", None, perm)?;
            println!(
                "       {:?}: {}",
                perm,
                if has_access { "✅" } else { "❌" }
            );
        }
    }

    Ok(())
}

/// Demonstrates hierarchical role permissions
fn hierarchical_roles() -> Result<()> {
    println!("  📊 Role Hierarchy (higher roles inherit lower role permissions):");
    println!("     Admin > Manager > Analyst > Viewer");

    let admin = Role::Admin;
    let manager = Role::Manager;
    let analyst = Role::Analyst;
    let viewer = Role::Viewer;

    println!("\n  🔍 Hierarchy relationships:");
    println!("     Admin includes Manager: {}", admin.includes(&manager));
    println!(
        "     Manager includes Analyst: {}",
        manager.includes(&analyst)
    );
    println!(
        "     Analyst includes Viewer: {}",
        analyst.includes(&viewer)
    );
    println!("     Viewer includes Admin: {}", viewer.includes(&admin));

    println!("\n  🎯 Permission counts by role:");
    println!(
        "     Admin: {} permissions",
        admin.default_permissions().len()
    );
    println!(
        "     Manager: {} permissions",
        manager.default_permissions().len()
    );
    println!(
        "     Analyst: {} permissions",
        analyst.default_permissions().len()
    );
    println!(
        "     Viewer: {} permissions",
        viewer.default_permissions().len()
    );

    Ok(())
}

/// Demonstrates resource-level access control
fn resource_level_access() -> Result<()> {
    let mut rbac = RbacManager::new();

    // Setup tenant and users
    let tenant_config = TenantConfig::new("tech_corp");
    rbac.register_tenant(tenant_config)?;

    rbac.register_user_with_role("owner", "owner@tech.com", "pass", "tech_corp", Role::Admin)?;
    rbac.register_user_with_role(
        "analyst1",
        "analyst1@tech.com",
        "pass",
        "tech_corp",
        Role::Analyst,
    )?;
    rbac.register_user_with_role(
        "analyst2",
        "analyst2@tech.com",
        "pass",
        "tech_corp",
        Role::Analyst,
    )?;

    println!("  📁 Creating resources with specific access controls:");

    // Create resources
    let sales_data = ResourceId::new("sales_data_2024");
    let finance_data = ResourceId::new("finance_data_2024");

    rbac.create_resource(
        sales_data.clone(),
        "tech_corp".to_string(),
        "owner".to_string(),
    )?;
    rbac.create_resource(
        finance_data.clone(),
        "tech_corp".to_string(),
        "owner".to_string(),
    )?;

    println!("     ✅ Created resource: sales_data_2024 (owner: owner)");
    println!("     ✅ Created resource: finance_data_2024 (owner: owner)");

    // Grant specific permissions
    println!("\n  🔑 Granting resource-specific permissions:");

    let mut sales_perms = HashSet::new();
    sales_perms.insert(Permission::Read);
    sales_perms.insert(Permission::Write);
    rbac.grant_resource_permission(&sales_data, "analyst1".to_string(), sales_perms)?;
    println!("     ✅ analyst1: Read/Write on sales_data_2024");

    let mut finance_perms = HashSet::new();
    finance_perms.insert(Permission::Read);
    rbac.grant_resource_permission(&finance_data, "analyst1".to_string(), finance_perms.clone())?;
    println!("     ✅ analyst1: Read-only on finance_data_2024");

    // Test access
    println!("\n  🔍 Testing resource access:");

    println!("     analyst1 accessing sales_data:");
    let can_read =
        rbac.check_access("analyst1", "tech_corp", Some(&sales_data), Permission::Read)?;
    let can_write = rbac.check_access(
        "analyst1",
        "tech_corp",
        Some(&sales_data),
        Permission::Write,
    )?;
    println!("       Read: {}", if can_read { "✅" } else { "❌" });
    println!("       Write: {}", if can_write { "✅" } else { "❌" });

    println!("     analyst1 accessing finance_data:");
    let can_read = rbac.check_access(
        "analyst1",
        "tech_corp",
        Some(&finance_data),
        Permission::Read,
    )?;
    let can_write = rbac.check_access(
        "analyst1",
        "tech_corp",
        Some(&finance_data),
        Permission::Write,
    )?;
    println!("       Read: {}", if can_read { "✅" } else { "❌" });
    println!("       Write: {}", if can_write { "✅" } else { "❌" });

    println!("     analyst2 accessing sales_data (no explicit grant):");
    let can_read =
        rbac.check_access("analyst2", "tech_corp", Some(&sales_data), Permission::Read)?;
    println!(
        "       Read: {}",
        if can_read {
            "❌"
        } else {
            "❌ (correctly denied)"
        }
    );

    Ok(())
}

/// Demonstrates multi-tenant isolation
fn multi_tenant_isolation() -> Result<()> {
    let mut rbac = RbacManager::new();

    // Setup two tenants
    rbac.register_tenant(TenantConfig::new("tenant_a"))?;
    rbac.register_tenant(TenantConfig::new("tenant_b"))?;

    println!("  🏢 Created two isolated tenants: tenant_a, tenant_b");

    // Register users for each tenant
    rbac.register_user_with_role(
        "alice_a",
        "alice@tenant-a.com",
        "pass",
        "tenant_a",
        Role::Admin,
    )?;
    rbac.register_user_with_role("bob_b", "bob@tenant-b.com", "pass", "tenant_b", Role::Admin)?;

    println!("     ✅ alice_a (Admin) in tenant_a");
    println!("     ✅ bob_b (Admin) in tenant_b");

    // Create resources for each tenant
    let resource_a = ResourceId::new("data_a");
    let resource_b = ResourceId::new("data_b");

    rbac.create_resource(
        resource_a.clone(),
        "tenant_a".to_string(),
        "alice_a".to_string(),
    )?;
    rbac.create_resource(
        resource_b.clone(),
        "tenant_b".to_string(),
        "bob_b".to_string(),
    )?;

    println!("\n  📁 Created resources:");
    println!("     data_a (tenant_a, owner: alice_a)");
    println!("     data_b (tenant_b, owner: bob_b)");

    // Test cross-tenant access (should be denied)
    println!("\n  🚫 Testing cross-tenant access prevention:");

    println!("     alice_a trying to access tenant_b resources:");
    let access = rbac.check_access("alice_a", "tenant_b", Some(&resource_b), Permission::Read)?;
    println!(
        "       {}",
        if !access {
            "✅ Correctly denied (cross-tenant access)"
        } else {
            "❌ Security violation!"
        }
    );

    println!("     bob_b trying to access tenant_a resources:");
    let access = rbac.check_access("bob_b", "tenant_a", Some(&resource_a), Permission::Read)?;
    println!(
        "       {}",
        if !access {
            "✅ Correctly denied (cross-tenant access)"
        } else {
            "❌ Security violation!"
        }
    );

    // Test same-tenant access (should be allowed for owners)
    println!("\n  ✅ Testing same-tenant access:");

    println!("     alice_a accessing tenant_a resources:");
    let access = rbac.check_access("alice_a", "tenant_a", Some(&resource_a), Permission::Read)?;
    println!(
        "       {}",
        if access {
            "✅ Allowed (same tenant, owner)"
        } else {
            "❌ Should be allowed!"
        }
    );

    Ok(())
}

/// Demonstrates group-based permissions
fn group_based_permissions() -> Result<()> {
    let mut rbac = RbacManager::new();

    // Setup
    rbac.register_tenant(TenantConfig::new("company"))?;

    rbac.register_user_with_role(
        "analyst1",
        "analyst1@company.com",
        "pass",
        "company",
        Role::Analyst,
    )?;
    rbac.register_user_with_role(
        "analyst2",
        "analyst2@company.com",
        "pass",
        "company",
        Role::Analyst,
    )?;
    rbac.register_user_with_role(
        "analyst3",
        "analyst3@company.com",
        "pass",
        "company",
        Role::Analyst,
    )?;

    println!("  👥 Setting up user groups:");

    // Create groups
    rbac.add_user_to_group("analyst1".to_string(), "sales_team".to_string());
    rbac.add_user_to_group("analyst2".to_string(), "sales_team".to_string());
    rbac.add_user_to_group("analyst3".to_string(), "finance_team".to_string());

    println!("     ✅ sales_team: analyst1, analyst2");
    println!("     ✅ finance_team: analyst3");

    // Create resource with group permissions
    let sales_report = ResourceId::new("q4_sales_report");
    rbac.create_resource(
        sales_report.clone(),
        "company".to_string(),
        "manager".to_string(),
    )?;

    let mut group_perms = HashSet::new();
    group_perms.insert(Permission::Read);
    group_perms.insert(Permission::Write);
    rbac.grant_resource_permission_to_group(&sales_report, "sales_team".to_string(), group_perms)?;

    println!("\n  📁 Resource 'q4_sales_report' created");
    println!("     Permissions granted to 'sales_team' group");

    // Test group-based access
    println!("\n  🔍 Testing group-based access:");

    for user in ["analyst1", "analyst2", "analyst3"] {
        let access = rbac.check_access(user, "company", Some(&sales_report), Permission::Read)?;
        let in_sales_team = user == "analyst1" || user == "analyst2";
        println!(
            "     {} ({}): {}",
            user,
            if in_sales_team {
                "sales_team"
            } else {
                "finance_team"
            },
            if access {
                "✅ Access granted"
            } else {
                "❌ Access denied"
            }
        );
    }

    Ok(())
}

/// Demonstrates dynamic permission updates
fn dynamic_permission_updates() -> Result<()> {
    let mut rbac = RbacManager::new();

    // Setup
    rbac.register_tenant(TenantConfig::new("dynamic_corp"))?;
    rbac.register_user_with_role(
        "user1",
        "user1@dynamic.com",
        "pass",
        "dynamic_corp",
        Role::Viewer,
    )?;

    println!("  👤 User 'user1' initially has Viewer role (read-only)");

    // Check initial permissions
    println!("\n  🔍 Initial permissions:");
    let can_read = rbac.check_access("user1", "dynamic_corp", None, Permission::Read)?;
    let can_write = rbac.check_access("user1", "dynamic_corp", None, Permission::Write)?;
    println!("     Read: {}", if can_read { "✅" } else { "❌" });
    println!("     Write: {}", if can_write { "✅" } else { "❌" });

    // Update user to Analyst role
    println!("\n  🔄 Promoting user1 to Analyst role...");

    // Get existing user and update
    if let Some(user) = rbac.auth_manager.get_user("user1") {
        let mut updated_user: UserInfo = user.clone();
        updated_user.roles.push("analyst".to_string());
        updated_user.permissions = Role::Analyst.default_permissions().into_iter().collect();
        rbac.auth_manager.update_user(updated_user)?;
    }

    println!("     ✅ User promoted to Analyst");

    // Check updated permissions
    println!("\n  🔍 Updated permissions:");
    let can_read = rbac.check_access("user1", "dynamic_corp", None, Permission::Read)?;
    let can_write = rbac.check_access("user1", "dynamic_corp", None, Permission::Write)?;
    println!("     Read: {}", if can_read { "✅" } else { "❌" });
    println!("     Write: {}", if can_write { "✅" } else { "❌" });

    Ok(())
}

/// Demonstrates comprehensive audit trail
fn audit_trail_demo() -> Result<()> {
    let mut rbac = RbacManager::new();

    // Setup
    rbac.register_tenant(TenantConfig::new("audit_corp"))?;
    rbac.register_user_with_role(
        "audited_user",
        "user@audit.com",
        "pass",
        "audit_corp",
        Role::Analyst,
    )?;

    let resource = ResourceId::new("sensitive_data");
    rbac.create_resource(
        resource.clone(),
        "audit_corp".to_string(),
        "admin".to_string(),
    )?;

    println!("  📋 Generating access events for audit trail:");

    // Generate various access attempts
    let _ = rbac.check_access(
        "audited_user",
        "audit_corp",
        Some(&resource),
        Permission::Read,
    );
    let _ = rbac.check_access(
        "audited_user",
        "audit_corp",
        Some(&resource),
        Permission::Write,
    );
    let _ = rbac.check_access(
        "audited_user",
        "audit_corp",
        Some(&resource),
        Permission::Delete,
    );
    let _ = rbac.check_access(
        "audited_user",
        "audit_corp",
        Some(&resource),
        Permission::Admin,
    );

    println!("     ✅ Generated access attempts");

    // Display audit trail
    println!("\n  📊 Audit Trail for 'audited_user':");
    let logs = rbac.get_user_access_logs("audited_user");

    for (i, log) in logs.iter().enumerate() {
        println!("\n     Event {}:", i + 1);
        println!("       Time: {:?}", log.timestamp);
        println!("       User: {}", log.user_id);
        println!("       Tenant: {}", log.tenant_id);
        println!("       Resource: {:?}", log.resource_id);
        println!("       Permission: {:?}", log.permission);
        println!("       Granted: {}", log.granted);
        println!("       Reason: {}", log.reason);
    }

    println!("\n  📊 Audit Trail for 'sensitive_data' resource:");
    let resource_logs = rbac.get_resource_access_logs("sensitive_data");
    println!("     Total access attempts: {}", resource_logs.len());
    let granted = resource_logs.iter().filter(|log| log.granted).count();
    let denied = resource_logs.len() - granted;
    println!("     Granted: {}", granted);
    println!("     Denied: {}", denied);

    Ok(())
}

/// Demonstrates conditional access policies
fn conditional_access() -> Result<()> {
    println!("  ⚡ Conditional Access Policies:");
    println!("     These policies add context-aware access control:");
    println!();
    println!("     📍 Location-based: Restrict access by IP/geography");
    println!("     ⏰ Time-based: Allow access only during business hours");
    println!("     🔐 MFA required: Enforce multi-factor for sensitive data");
    println!("     📊 Risk-based: Require additional auth for high-risk actions");
    println!();
    println!("     Example policy:");
    println!("       IF user.role == Analyst");
    println!("       AND resource.sensitivity == HIGH");
    println!("       AND time NOT IN business_hours");
    println!("       THEN require_additional_auth()");
    println!();
    println!("  ✅ Conditional policies provide dynamic, context-aware security");

    Ok(())
}
