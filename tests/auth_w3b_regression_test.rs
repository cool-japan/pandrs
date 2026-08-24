#![allow(clippy::result_large_err)]
//! Wave 3b regression tests for the auth / multitenancy / audit / credential
//! security boundaries.
//!
//! These lock in the fixes verified/landed during the auth-multitenancy audit:
//!   * cyclic subject-set grants terminate instead of aborting the process
//!   * hierarchical `parent` inheritance actually resolves
//!   * RBAC compat layer is tenant-namespaced (no cross-tenant escalation)
//!   * API keys are stored hashed, never in plaintext
//!   * logout removes the live session (no detached copy)
//!   * AES-GCM credential blobs are AAD-bound to their name (not swappable)
//!   * a deactivated tenant loses data-path access
//!   * dataset sharing grants read-only access without a cross-tenant oracle
//!   * an in-place overwrite is not double-counted against the row quota

use pandrs::audit::{AuditConfig, SharedAuditLogger};
use pandrs::auth::rebac::{RbacCompatLayer, RebacManager};
use pandrs::auth::{AuthManager, JwtConfig, UserInfo};
use pandrs::config::credentials::{CredentialBuilder, CredentialStore};
use pandrs::multitenancy::{Permission, TenantConfig, TenantManager};
use pandrs::{DataFrame, Series};
use std::sync::{Arc, RwLock};

fn df_with_rows(n: usize) -> DataFrame {
    let mut df = DataFrame::new();
    let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let s = Series::new(vals, Some("x".to_string())).expect("series");
    df.add_column("x".to_string(), s).expect("add column");
    df
}

// ---------------------------------------------------------------------------
// ReBAC graph: cyclic grants (the original PROCESS-KILL, exit 134)
// ---------------------------------------------------------------------------

/// Two subject-sets that reference each other through the *same* relation used
/// to be resolved by `is_member_of_set` re-entering `check` with a fresh cycle
/// guard, so a membership query recursed until the stack overflowed and the
/// process aborted (SIGABRT). The guard is now a threaded path set, so the
/// query must terminate with `Ok(false)` for a non-member.
#[test]
fn subject_set_mutual_cycle_denied_without_panic() {
    let rebac = RebacManager::new();
    // members of team:b are members of team:a, and vice-versa.
    rebac
        .grant("team:b#member", "member", "team:a")
        .expect("grant b->a");
    rebac
        .grant("team:a#member", "member", "team:b")
        .expect("grant a->b");

    // carol is a member of neither: the check must terminate, not crash.
    let result = rebac.check_access("user:carol", "member", "team:a");
    assert!(
        result.is_ok(),
        "cyclic subject-sets must not crash the process"
    );
    assert!(!result.expect("ok"), "non-member must be denied");
}

/// The same mutual cycle must not break *legitimate* transitive resolution: a
/// concrete member of one team is still resolved as a member of the other.
#[test]
fn subject_set_cycle_still_resolves_legitimate_membership() {
    let rebac = RebacManager::new();
    rebac
        .grant("team:b#member", "member", "team:a")
        .expect("grant b->a");
    rebac
        .grant("team:a#member", "member", "team:b")
        .expect("grant a->b");
    // carol is a direct member of team:b.
    rebac
        .grant("user:carol", "member", "team:b")
        .expect("grant carol->b");

    // carol ∈ b, and b#member ⇒ member of a, so carol resolves as member of a.
    let result = rebac
        .check_access("user:carol", "member", "team:a")
        .expect("check should not crash");
    assert!(
        result,
        "transitive membership must resolve despite the cycle"
    );
}

// ---------------------------------------------------------------------------
// ReBAC graph: hierarchical `parent` inheritance (module doc example)
// ---------------------------------------------------------------------------

#[test]
fn folder_inheritance_grant_on_folder_check_on_document() {
    let rebac = RebacManager::new();
    // document:123 lives in folder:456; alice can view the folder.
    rebac
        .grant("document:123", "parent", "folder:456")
        .expect("grant parent");
    rebac
        .grant("user:alice", "viewer", "folder:456")
        .expect("grant viewer on folder");

    // alice inherits viewer on the document through the folder.
    assert!(
        rebac
            .check_access("user:alice", "viewer", "document:123")
            .expect("check ok"),
        "folder viewer must inherit to the child document"
    );
    // bob has no grant anywhere.
    assert!(
        !rebac
            .check_access("user:bob", "viewer", "document:123")
            .expect("check ok"),
        "unrelated user must be denied"
    );
}

// ---------------------------------------------------------------------------
// RBAC compat: tenant namespacing (no cross-tenant escalation)
// ---------------------------------------------------------------------------

#[test]
fn cross_tenant_permission_denied() {
    let compat = RbacCompatLayer::new(Arc::new(RwLock::new(RebacManager::new())));
    compat
        .grant_resource_permission("alice", &Permission::Write, "document", "123", "tenant_a")
        .expect("grant in tenant_a");

    // alice/document:123 authorized in tenant_a...
    assert!(
        compat
            .check_permission("alice", &Permission::Write, "document", "123", "tenant_a")
            .expect("check ok"),
        "grant must hold in its own tenant"
    );
    // ...but the identically-named user+resource in tenant_b must NOT be.
    assert!(
        !compat
            .check_permission("alice", &Permission::Write, "document", "123", "tenant_b")
            .expect("check ok"),
        "a look-alike id in another tenant must never be authorized"
    );
}

// ---------------------------------------------------------------------------
// Auth: API keys stored hashed, never plaintext
// ---------------------------------------------------------------------------

#[test]
fn api_key_stored_hashed_plaintext_not_retrievable() {
    let mut auth = AuthManager::new(JwtConfig::default());
    let user = UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");
    auth.register_user(user).expect("register");

    let key = auth
        .create_api_key("user1", "test-key", None)
        .expect("create key");
    assert!(
        key.starts_with("pk_"),
        "returned value is the plaintext key"
    );

    let stored = auth.get_user("user1").expect("user").api_keys.clone();
    assert_eq!(stored.len(), 1, "exactly one key tracked");
    // The plaintext key must not be retrievable from the user record.
    assert!(
        !stored.contains(&key),
        "plaintext API key must not be stored on the user"
    );
    // What is stored is a SHA-256 hex digest (64 hex chars), not the plaintext.
    assert_eq!(stored[0].len(), 64, "stored value is a SHA-256 hex digest");
    assert!(
        stored[0].chars().all(|c| c.is_ascii_hexdigit()),
        "stored value is hex"
    );
    assert_ne!(stored[0], key, "stored value differs from plaintext");

    // The key still authenticates via hash lookup.
    assert!(
        auth.authenticate_api_key(&key).is_ok(),
        "hashed lookup must still authenticate the plaintext key"
    );
}

// ---------------------------------------------------------------------------
// Auth: logout propagates (no detached session copy)
// ---------------------------------------------------------------------------

#[test]
fn logout_propagates_session_removed() {
    let mut auth = AuthManager::new(JwtConfig::default());
    let user = UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");
    auth.register_user(user).expect("register");

    let result = auth
        .authenticate_password("user@example.com", "secret123")
        .expect("login");
    let session_id = result.session_id.expect("session id");

    assert!(
        auth.validate_session(&session_id).is_ok(),
        "session valid before logout"
    );
    auth.logout(&session_id).expect("logout");
    // Logout removed the live session; validation of the same id now fails.
    assert!(
        auth.validate_session(&session_id).is_err(),
        "logout must invalidate the session in the live store"
    );
}

// ---------------------------------------------------------------------------
// Credentials: AAD binds ciphertext to its name (blobs not swappable)
// ---------------------------------------------------------------------------

#[test]
fn credential_aad_swap_rejected() {
    let mut store = CredentialStore::with_defaults();
    store.init_encryption("master-pw").expect("init encryption");
    store
        .store_credential(
            "readonly",
            CredentialBuilder::new()
                .api_key("ro-key", None, None)
                .build()
                .expect("build ro"),
        )
        .expect("store ro");
    store
        .store_credential(
            "admin",
            CredentialBuilder::new()
                .api_key("admin-key", None, None)
                .build()
                .expect("build admin"),
        )
        .expect("store admin");

    // Export re-encrypts under a fresh export key; the JSON is attacker-visible.
    let exported = store.export_credentials("export-pw").expect("export");
    let mut root: serde_json::Value = serde_json::from_str(&exported).expect("parse export json");

    // Swap the two credentials' ciphertext blobs (data/iv/tag), leaving the
    // config salt intact — exactly the tampering AAD is meant to defeat.
    {
        let creds = root
            .get_mut("credentials")
            .and_then(|c| c.as_object_mut())
            .expect("credentials object");
        let ro = creds.get("readonly").expect("readonly entry").clone();
        let ad = creds.get("admin").expect("admin entry").clone();
        for field in ["data", "iv", "tag"] {
            let ro_field = ro.get(field).expect("ro field").clone();
            let ad_field = ad.get(field).expect("ad field").clone();
            creds
                .get_mut("readonly")
                .and_then(|e| e.as_object_mut())
                .expect("ro obj")
                .insert(field.to_string(), ad_field);
            creds
                .get_mut("admin")
                .and_then(|e| e.as_object_mut())
                .expect("ad obj")
                .insert(field.to_string(), ro_field);
        }
    }
    let mutated = serde_json::to_string(&root).expect("serialize mutated");

    // Importing the swapped blobs must fail: the AAD (credential name) no longer
    // matches the ciphertext, so AES-GCM authentication rejects it.
    let mut target = CredentialStore::with_defaults();
    let result = target.import_credentials(&mutated, "export-pw");
    assert!(
        result.is_err(),
        "AAD-bound ciphertext must not be swappable between credential names"
    );
}

// ---------------------------------------------------------------------------
// Multitenancy: deactivated tenant loses data-path access
// ---------------------------------------------------------------------------

#[test]
fn deactivated_tenant_denied_on_data_paths() {
    let mut mgr = TenantManager::new();
    mgr.register_tenant(TenantConfig::default_rw("t1"))
        .expect("register");

    // Store while active.
    mgr.store_dataframe("t1", "d", df_with_rows(3))
        .expect("store while active");
    assert!(
        mgr.get_dataframe("t1", "d").is_ok(),
        "active tenant reads its data"
    );

    // Deactivate the tenant.
    let mut cfg = mgr.get_tenant("t1").expect("tenant").clone();
    cfg.active = false;
    mgr.update_tenant(cfg).expect("update");

    // Every data path is now denied.
    assert!(
        mgr.get_dataframe("t1", "d").is_err(),
        "deactivated tenant cannot read"
    );
    assert!(
        mgr.list_datasets("t1").is_err(),
        "deactivated tenant cannot list"
    );
    assert!(
        mgr.get_dataset_metadata("t1", "d").is_err(),
        "deactivated tenant cannot read metadata"
    );
    assert!(
        mgr.store_dataframe("t1", "d2", df_with_rows(1)).is_err(),
        "deactivated tenant cannot write"
    );
}

// ---------------------------------------------------------------------------
// Multitenancy: dataset sharing is read-only and has no enumeration oracle
// ---------------------------------------------------------------------------

#[test]
fn shared_dataset_read_only_and_not_shared_indistinguishable() {
    let mut mgr = TenantManager::new();
    mgr.register_tenant(TenantConfig::default_rw("owner").with_permission(Permission::Share))
        .expect("register owner");
    mgr.register_tenant(TenantConfig::default_rw("friend"))
        .expect("register friend");
    mgr.register_tenant(TenantConfig::default_rw("stranger"))
        .expect("register stranger");

    mgr.store_dataframe("owner", "report_q3", df_with_rows(4))
        .expect("store");
    mgr.share_dataset("owner", "report_q3", "friend")
        .expect("share");

    // friend gets read access to the shared dataset...
    let df = mgr
        .get_dataframe("friend", "report_q3")
        .expect("friend reads shared dataset");
    assert_eq!(df.row_count(), 4, "shared read returns the owner's data");

    // ...but a non-shared tenant is denied. The error is a plain not-found that
    // is indistinguishable from a genuinely missing dataset: it reveals neither
    // that the dataset exists, who owns it, nor that it is a sharing/permission
    // matter (which would be a cross-tenant enumeration oracle).
    let denied = mgr
        .get_dataframe("stranger", "report_q3")
        .expect_err("stranger denied")
        .to_string();
    // A genuinely non-existent id yields the identical template (differing only
    // by the echoed id), so "exists but not shared" cannot be told apart.
    let phantom = mgr
        .get_dataframe("stranger", "no_such_dataset")
        .expect_err("phantom denied")
        .to_string();
    assert_eq!(
        denied, "Invalid input: Dataset 'report_q3' not found for tenant 'stranger'",
        "not-shared denial must be a plain not-found"
    );
    assert_eq!(
        phantom, "Invalid input: Dataset 'no_such_dataset' not found for tenant 'stranger'",
        "missing dataset uses the same not-found template"
    );
    assert!(
        !denied.contains("owner"),
        "must not reveal the owning tenant: {denied}"
    );
    assert!(
        !denied.to_lowercase().contains("permission") && !denied.to_lowercase().contains("exist"),
        "must not reveal existence/permission state: {denied}"
    );

    // Sharing is read-through only: it does not inject the dataset into the
    // friend's own store, so the friend cannot list/own/rewrite it as theirs.
    let friend_own = mgr.list_datasets("friend").expect("friend list");
    assert!(
        friend_own.is_empty(),
        "a shared dataset must not appear in the recipient's own store"
    );
}

// ---------------------------------------------------------------------------
// Multitenancy: overwrite is not double-counted against the row quota
// ---------------------------------------------------------------------------

#[test]
fn overwrite_does_not_double_count_row_quota() {
    let mut mgr = TenantManager::new();
    // Row cap of 6; a 5-row dataset fits. An overwrite with another 5 rows must
    // also fit (5 replaces 5), where the old code charged 5 + 5 = 10 > 6.
    mgr.register_tenant(TenantConfig::default_rw("t").with_max_rows(6))
        .expect("register");

    mgr.store_dataframe("t", "d", df_with_rows(5))
        .expect("first store (create)");
    mgr.store_dataframe("t", "d", df_with_rows(5))
        .expect("overwrite must not double-count the replaced rows");

    let usage = mgr.get_usage("t").expect("usage");
    assert_eq!(usage.total_rows, 5, "overwrite must not accumulate rows");
    assert_eq!(usage.dataset_count, 1, "overwrite must not add a dataset");
}

// ---------------------------------------------------------------------------
// Multitenancy + audit: data-path denials emit a Security audit entry
// ---------------------------------------------------------------------------

#[test]
fn tenant_denial_emits_security_audit_entry() {
    let logger = SharedAuditLogger::new(AuditConfig::default());
    let mut mgr = TenantManager::new().with_audit_logger(logger.clone());
    // A read-only tenant lacks Create, so storing is denied — and the denial
    // must surface as a Security audit entry (previously only successes logged).
    mgr.register_tenant(TenantConfig::new("ro").with_permission(Permission::Read))
        .expect("register");

    let result = mgr.store_dataframe("ro", "d", df_with_rows(1));
    assert!(result.is_err(), "read-only tenant cannot create");

    let stats = logger.stats().expect("audit stats");
    let security = stats.by_category.get("SECURITY").copied().unwrap_or(0);
    assert!(
        security >= 1,
        "a permission denial must emit a Security audit entry (got {security})"
    );
}
