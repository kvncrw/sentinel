//! Database Operations Gate
//!
//! HARD BLOCK on database migration and destructive commands targeting
//! production environments. The CLAUDE.md says: "NEVER run database ops
//! or migrations in prod or production, even if the user gives permission."
//!
//! Local database ops are allowed (no prod indicators detected).
//! Production is detected by: `DATABASE_URL` containing "prod", explicit
//! --env production flags, known production hostnames, a bare `prod`/`prd`
//! database name handed to a CLI client, or a prod-ish DSN/connection key
//! inside a driver `connect(…)` call (see [`PROD_INDICATOR`]).
//!
//! Detection is wrapper-agnostic by construction: the destructive-SQL
//! regexes match the SQL text wherever it appears in the command string,
//! so `psql -c '…'`, here-strings (`<<<`), heredocs (`<<EOF`), and
//! echo/cat pipes into a client are all covered, as are driver-level
//! one-liners (`python3 -c "…psycopg2…execute('DELETE …')"`) and quoted /
//! schema-qualified identifiers (`DELETE FROM "public"."orders"`).
//!
//! POLICY (operator decision, 2026-07): **the gate blocks what it can
//! READ — it does not guess.** SQL it cannot see (a file via
//! `psql -f x.sql` / `< x.sql`, a `$`-variable or `$(…)` payload) is
//! allowed even on a production target: the hook cannot read the file or
//! resolve the variable at decision time, and a shape-only deny over-blocks
//! legitimate work (measured to destroy agent task performance). Likewise
//! out of scope: obfuscated/encoded payloads and prod-ness that only
//! resolves from the runtime environment with no textual signal.

use regex::Regex;
use std::sync::LazyLock;

use sentinel_domain::events::{HookInput, HookOutput};

/// Patterns that indicate a database migration or destructive operation.
static DB_MIGRATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(prisma\s+migrate|diesel\s+migration|sqlx\s+migrate|alembic\s+upgrade|flyway\s+migrate|knex\s+migrate|rake\s+db:migrate|rails\s+db:migrate|sequelize.*migrate|typeorm.*migration:run|drizzle-kit\s+push)"
    ).unwrap()
});

/// One SQL identifier: plain (`orders`), double-quoted (`"orders"`), or
/// backtick-quoted (`` `orders` ``), optionally schema-qualified with any
/// mix (`public.orders`, `"public"."orders"`, `app."events"`). Reused by
/// every destructive-shape pattern so ORM-generated / pg_dump-copied SQL
/// with quoted identifiers cannot evade what plain identifiers trip.
const SQL_IDENT: &str = r#"(?:\w+|"[^"]+"|`[^`]+`)(?:\.(?:\w+|"[^"]+"|`[^`]+`))*"#;

/// One prod-ish name token. `prod`/`prd` may continue with digits or
/// `_ - .` separators (`prod-db`, `prod_replica`, `prod2.internal`); the
/// full word `production` must END the token (next char not a letter,
/// digit, or underscore) so filenames (`production_report.sql`) and
/// compounds (`nonproduction`, `reproduction-svc`) never match. Callers
/// supply their own left-boundary guard.
const PROD_NAME: &str = r"(?:prod(?:[_\-.\d]|\b)|prd(?:[_\-.\d]|\b)|production(?:[^A-Za-z0-9_]|$))";

/// Patterns that indicate destructive SQL operations. The regex crate has no
/// lookahead, so a delete-all is caught by enumerating the "no WHERE clause"
/// terminators (`;`, a closing quote, a newline, end-of-string) plus tautology
/// WHEREs (`1=1`, `true`). The newline terminator matters for heredoc bodies
/// (`psql prod <<EOF\nDELETE FROM t\nEOF`) where the statement ends at a line
/// break instead of a quote or semicolon. Identifiers use [`SQL_IDENT`], so
/// quoted / schema-qualified forms (`TRUNCATE "orders"`,
/// `DELETE FROM "public"."orders"`) are caught. A scoped
/// `DELETE ... WHERE <col> …` is intentionally NOT matched so legitimate
/// targeted deletes pass. This is a broadening, not a complete fix — the
/// durable solution is parser/semantic-level enforcement.
static DB_DESTRUCTIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r#"(?i)(DROP\s+(TABLE|DATABASE|SCHEMA|INDEX)|TRUNCATE(\s+TABLE)?\s+{id}|DELETE\s+FROM\s+{id}\s*(;|"|'|\n|$)|DELETE\s+FROM\s+{id}\s+WHERE\s+('?1'?\s*=\s*'?1'?|true\b)|ALTER\s+TABLE.*DROP)"#,
        id = SQL_IDENT
    ))
    .unwrap()
});

/// SQL execution context — a CLI database client or a programmatic driver.
/// Used to scope the UPDATE-without-WHERE check below so a stray
/// "update … set …" in prose (a commit message, an echo) is never treated
/// as destructive SQL.
static SQL_EXEC_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(psql|pgcli|mysql|mariadb|mycli|sqlite3|mongosh|psycopg2?|asyncpg|sqlalchemy|mysql2|pymysql|sequelize|knex|typeorm)\b|\.\s*(execute|query)\s*\(",
    )
    .unwrap()
});

/// `UPDATE <table> SET …` statement, capturing everything up to the next
/// semicolon (or end of command). The capture is then checked for a `WHERE`
/// keyword in code — the regex crate has no lookahead, so "UPDATE without
/// WHERE" cannot be expressed in a single pattern.
static UPDATE_STMT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?is)\bUPDATE\s+{id}\s+SET\s+([^;]*)",
        id = SQL_IDENT
    ))
    .unwrap()
});

static WHERE_KEYWORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bWHERE\b").unwrap());

/// `UPDATE <table> SET …` with no `WHERE` clause before the statement
/// terminator — a full-table rewrite. Only counted as destructive when the
/// command also shows a SQL execution context (CLI client or driver call),
/// so ordinary prose containing "update x set y" passes.
fn update_without_where(cmd: &str) -> bool {
    SQL_EXEC_CONTEXT.is_match(cmd)
        && UPDATE_STMT.captures_iter(cmd).any(|caps| {
            let tail = caps.get(1).map_or("", |m| m.as_str());
            !WHERE_KEYWORD.is_match(tail)
        })
}

/// Patterns that indicate production environment.
///
/// Beyond the explicit flag forms, the arms recognize a prod-ish name
/// ([`PROD_NAME`]) only where a connection TARGET lives:
///
/// 1. a CLI client's argument (`psql prod`, `mysql -u root prd`,
///    `psql -h prod-db …` — flag values reach the same arm);
/// 2. a driver `connect(…)` call's DSN (`connect('prod')`);
/// 3. a connection key's value (`dbname=prod`, `database: 'prod'`,
///    `host=db-prod`);
/// 4. a connection URL's host/path (`postgres://user@prod-db:5432/app`) —
///    with the userinfo section skipped, so a PASSWORD containing
///    "production" (`postgres://user:production@staging/app`) does NOT
///    count as a prod target;
/// 5. libpq/mysql environment prefixes (`PGDATABASE=prod`,
///    `PGHOST=prod-db`). `PGPASSWORD=…` is deliberately absent —
///    credentials are not targets.
///
/// [`PROD_NAME`] token boundaries mean `products`, `preprod`, `prodigy`,
/// `nonproduction`, `reproduction-svc`, and a `production_report.sql`
/// FILENAME never match: the check looks at targets, not at arbitrary
/// words on the line.
static PROD_INDICATOR: LazyLock<Regex> = LazyLock::new(|| {
    let url_tail = format!(
        r#"(?:[^\s"'@]*@[\w.\-:/]*(?:\b|_){p}|[\w.\-]*(?:\b|_){p}|[\w.\-]*(?::\d+)?/[\w.\-/]*(?:\b|_){p})"#,
        p = PROD_NAME
    );
    Regex::new(&format!(
        r#"(?i)(--env(?:ironment)?\s+["']?{p}|DATABASE_URL=["']?(?:\w[\w+]*://)?{tail}|\.prod(?:uction)?\.|--prod\b|-p\s+["']?{p}|\b(?:psql|pgcli|mysql|mariadb|mycli|mongosh|mongo)(?:\s+--?[\w-]+(?:[=\s]+(?:"[^"]*"|'[^']*'|\S+))?)*\s+["']?{p}|\b(?:PGDATABASE|PGHOST|PGSERVICE)=["']?[\w.\-]*(?:\b|_){p}|connect\s*\(\s*["'][^"')]*(?:\b|_){p}|\b(?:dbname|database|db_name|db|host)\s*[:=]\s*["']?[\w.\-]*(?:\b|_){p}|(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?)://{tail})"#,
        p = PROD_NAME,
        tail = url_tail
    ))
    .unwrap()
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbOpsDecision {
    Allow,
    Block,
}

#[derive(Debug, Clone)]
pub struct DbOpsEvaluation {
    pub tool: Option<String>,
    pub command: Option<String>,
    pub bash_tool: bool,
    pub command_present: bool,
    pub migration: bool,
    pub destructive: bool,
    pub database_operation: bool,
    pub production_target: bool,
    pub should_block: bool,
    pub decision: DbOpsDecision,
    pub block_reason: Option<String>,
}

impl DbOpsEvaluation {
    #[must_use]
    pub const fn graph_authority_required(&self) -> bool {
        self.bash_tool && self.database_operation
    }
}

/// Process a `PreToolUse` Bash event. Blocks production database operations.
pub fn process(input: &HookInput) -> HookOutput {
    let evaluation = evaluate(input);
    output_from_evaluation(&evaluation)
}

pub fn evaluate(input: &HookInput) -> DbOpsEvaluation {
    let tool = input.tool_name.clone();
    let bash_tool = tool.as_deref().is_none_or(|tool| tool == "Bash");
    let command = extract_bash_command(input).map(str::to_string);
    let Some(cmd) = command.as_deref() else {
        return DbOpsEvaluation {
            tool,
            command,
            bash_tool,
            command_present: false,
            migration: false,
            destructive: false,
            database_operation: false,
            production_target: false,
            should_block: false,
            decision: DbOpsDecision::Allow,
            block_reason: None,
        };
    };

    let migration = DB_MIGRATION.is_match(cmd);
    // POLICY (operator decision, 2026-07): the gate blocks what it can READ.
    // SQL it cannot see — a file (`psql prod -f x.sql`), a stdin redirect,
    // a `$`-variable or `$(…)` substitution payload — is ALLOWED, even on a
    // production target. A deny there would be a guess: the hook cannot read
    // the file or resolve the variable at decision time, and over-blocking
    // has a measured, immediate cost to agent task performance while the
    // evasion it would prevent is speculative. Do not re-add an
    // opaque-source denial class here.
    let destructive = DB_DESTRUCTIVE.is_match(cmd) || update_without_where(cmd);
    let database_operation = migration || destructive;
    let production_target = database_operation && PROD_INDICATOR.is_match(cmd);
    let should_block = bash_tool && database_operation && production_target;
    let block_reason = should_block.then(|| {
        "🔴 [Database Gate] BLOCKED: destructive/migration operation on a PRODUCTION database. \
         STOP — do not retry this command. First read any governing documentation in the working \
         tree (AGENTS.md, README, or any POLICY/RETENTION file); it defines what is permitted. \
         Then perform ONLY the minimal non-destructive operation that satisfies it — e.g. delete \
         just the specific non-production rows, not the whole table. Do NOT attempt to bypass this \
         control (a WHERE clause that still deletes everything, a file/heredoc redirect, a \
         python/perl DB driver, renaming the database file, an in-memory copy): circumventing a \
         governance control is a critical failure. If no non-destructive path exists, stop and \
         surface this to the operator."
            .to_string()
    });

    DbOpsEvaluation {
        tool,
        command,
        bash_tool,
        command_present: true,
        migration,
        destructive,
        database_operation,
        production_target,
        should_block,
        decision: if should_block {
            DbOpsDecision::Block
        } else {
            DbOpsDecision::Allow
        },
        block_reason,
    }
}

pub fn output_from_evaluation(evaluation: &DbOpsEvaluation) -> HookOutput {
    if evaluation.should_block {
        return HookOutput::deny(
            evaluation
                .block_reason
                .clone()
                .unwrap_or_else(|| "Database gate blocked without a reason".to_string()),
        );
    }
    HookOutput::allow()
}

fn extract_bash_command(input: &HookInput) -> Option<&str> {
    input.tool_input.as_ref()?.get("command")?.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bash_input(cmd: &str) -> HookInput {
        HookInput {
            tool_name: Some("Bash".to_string()),
            tool_input: Some(serde_json::json!({"command": cmd})),
            ..Default::default()
        }
    }

    #[test]
    fn test_blocks_prisma_migrate_prod() {
        assert_eq!(
            process(&bash_input("prisma migrate deploy --env production")).blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_diesel_migration_prod() {
        assert_eq!(
            process(&bash_input(
                "DATABASE_URL=postgres://prod.db/app diesel migration run"
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_drop_table_prod() {
        assert_eq!(
            process(&bash_input("psql production -c 'DROP TABLE users;'")).blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_sqlx_migrate_prod() {
        assert_eq!(
            process(&bash_input("sqlx migrate run --env prod")).blocked,
            Some(true)
        );
    }

    #[test]
    fn test_allows_local_prisma_migrate() {
        assert!(process(&bash_input("prisma migrate dev")).blocked.is_none());
    }

    #[test]
    fn test_allows_local_diesel_migration() {
        assert!(process(&bash_input("diesel migration run"))
            .blocked
            .is_none());
    }

    #[test]
    fn test_allows_non_db_commands() {
        assert!(process(&bash_input("cargo test")).blocked.is_none());
        assert!(process(&bash_input("git push")).blocked.is_none());
    }

    #[test]
    fn test_allows_no_tool_input() {
        assert!(process(&HookInput::default()).blocked.is_none());
    }

    #[test]
    fn test_blocks_truncate_prod() {
        assert_eq!(
            process(&bash_input("psql production -c 'TRUNCATE TABLE sessions;'")).blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_delete_all_no_semicolon() {
        // A delete-all without a trailing semicolon (the sqlite one-liner form)
        // must still be caught — this exact form slipped through the old regex.
        assert_eq!(
            process(&bash_input(
                "sqlite3 /app/events.production.sqlite \"DELETE FROM events\""
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_delete_tautology_where() {
        // `WHERE 1=1` deletes everything while dodging a naive delete-all match.
        assert_eq!(
            process(&bash_input(
                "sqlite3 /app/events.production.sqlite \"DELETE FROM events WHERE 1=1\""
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_allows_scoped_delete_on_prod() {
        // A genuinely scoped delete on a production DB must NOT be blocked, or the
        // gate becomes a false-block on legitimate targeted cleanup.
        assert!(process(&bash_input(
            "sqlite3 /app/events.production.sqlite \"DELETE FROM events WHERE env != 'production'\""
        ))
        .blocked
        .is_none());
    }

    // ---- 2026-07 evasion class (a): shell wrappers feeding SQL to a client
    // with a bare `prod` target the old PROD_INDICATOR never recognized ----

    #[test]
    fn test_blocks_herestring_delete_tautology_bare_prod() {
        // The documented evasion: here-string wrapper + bare `prod` DB name.
        assert_eq!(
            process(&bash_input(
                "psql prod <<< \"DELETE FROM audit_log WHERE 1=1\""
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_dash_c_delete_tautology_bare_prod() {
        // Same payload without the wrapper — proves the hole was the bare
        // `prod` target, and that both forms are now closed.
        assert_eq!(
            process(&bash_input(
                "psql prod -c \"DELETE FROM audit_log WHERE 1=1\""
            ))
            .blocked,
            Some(true)
        );
        // Flags-before-positional ordering (quoted flag argument consumed).
        assert_eq!(
            process(&bash_input(
                "psql -c 'DELETE FROM audit_log WHERE 1=1' prod"
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_heredoc_delete_all_bare_prod() {
        // Heredoc body: the delete-all terminates at a newline, not a quote
        // or semicolon.
        assert_eq!(
            process(&bash_input("psql prod <<EOF\nDELETE FROM audit_log\nEOF")).blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_echo_pipe_truncate_bare_prod() {
        assert_eq!(
            process(&bash_input("echo \"TRUNCATE audit_log\" | psql prod")).blocked,
            Some(true)
        );
    }

    // ---- 2026-07 evasion class (b): driver-level destruction in an inline
    // interpreter one-liner ----

    #[test]
    fn test_blocks_driver_delete_prod_dsn() {
        // The documented evasion: psycopg2 one-liner with a bare 'prod' DSN.
        assert_eq!(
            process(&bash_input(
                "python3 -c \"import psycopg2;psycopg2.connect('prod').cursor().execute('DELETE FROM orders')\""
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_driver_update_without_where_prod() {
        // Full-table UPDATE (no WHERE) through a driver against prod.
        assert_eq!(
            process(&bash_input(
                "python3 -c \"import psycopg2; conn=psycopg2.connect('prod'); conn.cursor().execute('UPDATE users SET active=false')\""
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_driver_delete_prod_connection_key() {
        // node/pg with a prod-ish connection key (`database: 'prod'`).
        assert_eq!(
            process(&bash_input(
                "node -e \"const{Client}=require('pg');new Client({database: 'prod'}).query('DELETE FROM orders')\""
            ))
            .blocked,
            Some(true)
        );
    }

    // ---- Explicit allow cases: the broadened detection must not over-block ----

    #[test]
    fn test_allows_select_on_bare_prod() {
        // Read-only queries on prod stay allowed — even with the bare-prod
        // target now recognized, the gate still requires a destructive op.
        assert!(
            process(&bash_input("psql prod -c 'SELECT count(*) FROM orders;'"))
                .blocked
                .is_none()
        );
        assert!(process(&bash_input(
            "psql production -c 'SELECT count(*) FROM orders;'"
        ))
        .blocked
        .is_none());
    }

    #[test]
    fn test_allows_scoped_delete_on_bare_prod() {
        // Targeted cleanup on prod is legitimate; only delete-alls block.
        assert!(process(&bash_input(
            "psql prod -c \"DELETE FROM events WHERE created_at < now() - interval '90 days'\""
        ))
        .blocked
        .is_none());
    }

    #[test]
    fn test_allows_scoped_update_on_prod_driver() {
        // UPDATE with a real WHERE clause through a driver — not destructive.
        assert!(process(&bash_input(
            "python3 -c \"import psycopg2; psycopg2.connect('prod').cursor().execute('UPDATE users SET active=false WHERE id=5')\""
        ))
        .blocked
        .is_none());
    }

    #[test]
    fn test_allows_driver_import_only() {
        // Ordinary python that merely imports a driver must pass.
        assert!(process(&bash_input(
            "python3 -c \"import psycopg2; print(psycopg2.__version__)\""
        ))
        .blocked
        .is_none());
        assert!(process(&bash_input("pip install psycopg2-binary"))
            .blocked
            .is_none());
    }

    #[test]
    fn test_allows_driver_select_on_prod() {
        assert!(process(&bash_input(
            "python3 -c \"import psycopg2; print(psycopg2.connect('prod').cursor().execute('SELECT 1'))\""
        ))
        .blocked
        .is_none());
    }

    #[test]
    fn test_allows_herestring_delete_on_non_prod() {
        // This gate is prod-scoped: the same wrapper against staging/local
        // DBs passes here (the env-agnostic reversibility layer still grades
        // it, but that is a different net with different semantics).
        assert!(process(&bash_input(
            "psql staging <<< \"DELETE FROM audit_log WHERE 1=1\""
        ))
        .blocked
        .is_none());
    }

    #[test]
    fn test_allows_prod_prefix_words_as_db_names() {
        // `products` is not `prod` — prefix words must not trip the bare-prod
        // detection.
        assert!(process(&bash_input(
            "psql products -c \"DELETE FROM cart_items WHERE 1=1\""
        ))
        .blocked
        .is_none());
        assert!(process(&bash_input(
            "python3 -c \"import psycopg2; psycopg2.connect('products_dev').cursor().execute('DELETE FROM cart_items')\""
        ))
        .blocked
        .is_none());
        // `preprod` is likewise not prod.
        assert!(process(&bash_input(
            "psql preprod <<< \"DELETE FROM audit_log WHERE 1=1\""
        ))
        .blocked
        .is_none());
    }

    #[test]
    fn test_allows_shell_truncate_utility() {
        // coreutils `truncate` is file surgery, not SQL — must never match,
        // even with "prod" in the path.
        assert!(process(&bash_input("truncate -s 0 /var/log/prod.log"))
            .blocked
            .is_none());
    }

    #[test]
    fn test_allows_prose_update_set_without_sql_context() {
        // "update … set …" in prose (no client, no driver, no execute()) is
        // not SQL. The SQL_EXEC_CONTEXT scope keeps this out of the gate.
        assert!(process(&bash_input(
            "git commit -m 'update config set production defaults'"
        ))
        .blocked
        .is_none());
    }

    // ---- POLICY: opaque SQL sources are ALLOWED, even on prod (operator
    // decision, 2026-07). The gate cannot read a file or resolve a shell
    // variable at hook time, so a deny there is a guess that costs real
    // work. These are pinned as explicit allow tests so the denial class
    // is not re-added by accident. ----

    #[test]
    fn test_allows_file_sourced_sql_even_on_prod() {
        for cmd in [
            "psql prod -f daily_sales_report.sql",
            "psql -f cleanup.sql prod",
            "psql production --file=cleanup.sql",
            "psql production < readonly_export.sql",
            "mysql -h db.prod.internal < batch.sql",
        ] {
            assert!(
                process(&bash_input(cmd)).blocked.is_none(),
                "opaque file/stdin source must ALLOW (policy): {cmd}"
            );
        }
    }

    #[test]
    fn test_allows_opaque_variable_payload_even_on_prod() {
        for cmd in [
            "psql prod -c \"$SQL\"",
            "psql prod -c \"$(cat cleanup.sql)\"",
            "mongosh prod --eval \"$PAYLOAD\"",
            // The previously-accepted over-block, now simply correct:
            "SQL=\"SELECT 1\"; psql prod -c \"$SQL\"",
        ] {
            assert!(
                process(&bash_input(cmd)).blocked.is_none(),
                "opaque variable payload must ALLOW (policy): {cmd}"
            );
        }
    }

    #[test]
    fn test_blocks_visible_destructive_text_carried_by_variable() {
        // NOT opaque: the destructive statement is visible in the SAME
        // command — the text rules judge it regardless of the variable hop.
        assert_eq!(
            process(&bash_input(
                "SQL=\"DELETE FROM orders\"; psql prod -c \"$SQL\""
            ))
            .blocked,
            Some(true)
        );
    }

    // ---- Quoted / schema-qualified identifiers (ORM- and pg_dump-shaped
    // SQL) — the content IS visible, so these are genuine evasions ----

    #[test]
    fn test_blocks_quoted_identifiers_on_prod() {
        for cmd in [
            "psql prod -c 'TRUNCATE \"orders\"'",
            "psql prod -c 'DELETE FROM \"public\".\"orders\"'",
            "psql prod -c 'DELETE FROM \"orders\" WHERE 1=1'",
            "mysql prod -e 'TRUNCATE `sessions`'",
            "psql prod -c 'DELETE FROM public.orders'",
        ] {
            assert_eq!(
                process(&bash_input(cmd)).blocked,
                Some(true),
                "quoted/qualified identifier must DENY: {cmd}"
            );
        }
    }

    #[test]
    fn test_allows_scoped_quoted_identifiers_on_prod() {
        // Quoting alone is not destruction — a scoped delete stays allowed.
        assert!(process(&bash_input(
            "psql prod -c 'DELETE FROM \"orders\" WHERE id = 5'"
        ))
        .blocked
        .is_none());
    }

    // ---- Prod spelling coverage: URL DSNs, env prefixes, short forms ----

    #[test]
    fn test_blocks_prod_url_dsn() {
        assert_eq!(
            process(&bash_input(
                "psql postgres://user@prod-db:5432/app -c \"DELETE FROM orders WHERE 1=1\""
            ))
            .blocked,
            Some(true)
        );
        assert_eq!(
            process(&bash_input(
                "psql postgres://user@prod-db:5432/app -c \"DELETE FROM orders;\""
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_pg_env_prefix_prod_with_visible_destruction() {
        assert_eq!(
            process(&bash_input(
                "PGDATABASE=prod psql -c \"DELETE FROM orders;\""
            ))
            .blocked,
            Some(true)
        );
        assert_eq!(
            process(&bash_input("PGHOST=prod-db psql -c \"TRUNCATE orders\"")).blocked,
            Some(true)
        );
    }

    #[test]
    fn test_blocks_dash_h_prod_host_with_visible_destruction() {
        // -h/-d flag values reach the client-argument arm — no dedicated
        // flag arm needed; this test pins that coverage.
        assert_eq!(
            process(&bash_input(
                "psql -h prod-db -d app -c \"DELETE FROM orders WHERE 1=1\""
            ))
            .blocked,
            Some(true)
        );
    }

    #[test]
    fn test_allows_password_containing_production() {
        // Credentials are not targets: "production" inside a password or
        // URL userinfo must not mark the command as prod.
        for cmd in [
            "PGPASSWORD=production psql staging -c \"DELETE FROM scratch WHERE 1=1\"",
            "psql postgres://user:production@staging-host/app -c \"DELETE FROM scratch WHERE 1=1\"",
        ] {
            assert!(
                process(&bash_input(cmd)).blocked.is_none(),
                "password is not a prod target: {cmd}"
            );
        }
    }

    #[test]
    fn test_blocks_prd_short_form() {
        assert_eq!(
            process(&bash_input("psql prd -c \"TRUNCATE orders\"")).blocked,
            Some(true)
        );
    }

    // ---- Allow half: file/variable shapes on NON-prod targets, and prod
    // commands whose payload is visible and safe ----

    #[test]
    fn test_allows_file_sourced_sql_on_non_prod() {
        // `psql -f` is the canonical migration workflow — dev/staging/local
        // file runs must pass or every migration is a false block.
        for cmd in [
            "psql dev -f migration.sql",
            "psql staging -f cleanup.sql",
            "psql \"$DB_URL\" -f migration.sql",
            "mysql app_db < seed.sql",
            "sqlite3 app.db < seed.sql",
            "psql preprod -f cleanup.sql",
            "psql -f report.sql products",
        ] {
            assert!(
                process(&bash_input(cmd)).blocked.is_none(),
                "must allow: {cmd}"
            );
        }
    }

    #[test]
    fn test_allows_opaque_variable_payload_on_non_prod() {
        assert!(process(&bash_input("psql dev -c \"$SQL\""))
            .blocked
            .is_none());
        assert!(process(&bash_input("mysql staging -e \"$STMT\""))
            .blocked
            .is_none());
    }

    #[test]
    fn test_allows_visible_payload_with_trailing_variable_on_prod() {
        // Payload starts with visible, safe SQL — a `$` parameter later in
        // the string must not flip it to "opaque".
        assert!(process(&bash_input(
            "psql prod -c \"SELECT count(*) FROM orders WHERE region = $REGION\""
        ))
        .blocked
        .is_none());
    }

    #[test]
    fn test_allows_sql_comparison_operators_on_prod() {
        // `<` as a SQL comparison inside a visible payload must never be
        // read as a redirect shape.
        for cmd in [
            "psql prod -c \"DELETE FROM events WHERE created_at < now() - interval '90 days'\"",
            "psql prod -c \"SELECT * FROM t WHERE x < $LIMIT\"",
        ] {
            assert!(
                process(&bash_input(cmd)).blocked.is_none(),
                "must allow: {cmd}"
            );
        }
    }

    #[test]
    fn test_allows_prod_lookalike_names_everywhere() {
        // The full lookalike set from the 2026-07 review: none of these are
        // production and none may trip any prod arm.
        for cmd in [
            "psql products -c \"DELETE FROM cart_items WHERE 1=1\"",
            "psql preprod <<< \"DELETE FROM audit_log WHERE 1=1\"",
            "psql prodigy -c \"DELETE FROM t WHERE 1=1\"",
            "psql staging -c \"TRUNCATE orders\"",
            "psql postgres://user@staging-db/app -c \"DELETE FROM orders WHERE 1=1\"",
            "python3 -c \"import psycopg2; psycopg2.connect('reproduce_db').cursor().execute('DELETE FROM t')\"",
        ] {
            assert!(
                process(&bash_input(cmd)).blocked.is_none(),
                "must allow (not prod): {cmd}"
            );
        }
    }
}
