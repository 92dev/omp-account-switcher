/**
 * `/accounts` — pick which stored OAuth account this session uses for a provider.
 *
 * omp already stores several OAuth credentials per provider (`auth_credentials`
 * rows, one per email/org/project) and routes each session to one of them via
 * the auth store's session-sticky selection. The built-in `/session pin` can
 * only re-pin the *current model's* provider and prints a bare numbered list.
 * This command covers every provider that has stored accounts, drives the
 * choice through the interactive selector, and annotates each account with its
 * last recorded quota usage so "which account has headroom" is answerable at
 * the moment of switching.
 *
 * `/accounts info` shows every stored OAuth account across all providers, with
 * session-active accounts marked and recorded quota usage alongside each row.
 * The view is read-only and remains available while streaming.
 *
 * The pin is session-scoped, exactly like `/session pin`: it sets the sticky
 * credential for `(provider, sessionId)`. Usage-limit handling and auth retry
 * may still route around a pinned account that is blocked or broken, and core
 * persists whichever account actually served a turn into the session file, so
 * a resume re-pins it on its own.
 */

import type { UsageHistoryEntry } from "@oh-my-pi/pi-ai";
import type {
	AuthStorage,
	ExtensionAPI,
	ExtensionCommandContext,
	OAuthAccountSummary,
} from "@oh-my-pi/pi-coding-agent";

/** How far back to look for a usage snapshot before calling an account "no data". */
const USAGE_LOOKBACK_MS = 24 * 60 * 60 * 1000;
/**
 * Limits of one account are written by one fetch, but providers report several
 * windows and a retry can straddle a minute boundary. Rows older than this
 * relative to the account's newest row belong to a previous snapshot and are
 * dropped so a stale window never gets rendered next to a fresh one.
 */
const USAGE_SNAPSHOT_SPREAD_MS = 15 * 60 * 1000;
/**
 * A live refresh is best-effort: `usage.reports` is per-credential cached
 * (5 min + jitter) so it is usually a no-op, but a cold cache means N provider
 * round-trips. Cap the wait and fall back to recorded history.
 */
const USAGE_REFRESH_TIMEOUT_MS = 2_500;
/** Windows rendered per account, most-used first. */
const USAGE_LIMIT_COLUMNS = 3;
/** Below this, a snapshot is "now" and its age is not worth a column. */
const USAGE_FRESH_AGE_MS = 60_000;

interface AccountUsage {
	/** Epoch ms of the snapshot these columns came from. */
	recordedAt: number;
	/** Rendered `"<window> <pct>%"` columns, most-used first. */
	columns: string[];
	/** Soonest future reset among the rendered windows. */
	resetsAt?: number;
}

interface Account extends OAuthAccountSummary {
	label: string;
	usage?: AccountUsage;
}

interface ProviderAccounts {
	provider: string;
	accounts: Account[];
}

/**
 * Base identifier plus the org/workspace that qualifies it. Two Anthropic
 * subscriptions can share one email and differ only by org, and that org is
 * the quota pool — dropping it would render two indistinguishable rows.
 */
function formatAccountLabel(account: OAuthAccountSummary): string {
	const base =
		account.email || account.accountId || account.projectId || account.enterpriseUrl || `credential #${account.credentialId}`;
	const org = account.orgName || account.orgId;
	return org && org !== base ? `${base} (${org})` : base;
}

/**
 * Rebuild the key `AuthStorage` records usage history under
 * (`#buildUsageCacheIdentity`): `oauth|account:<id>|email:<lower>|org:<id>|project:<id>|enterprise:<lower>`,
 * each segment present only when the credential carries it. Matching on the
 * whole key keeps same-email/different-org accounts apart, which a plain email
 * or accountId comparison cannot do.
 */
function usageIdentityKey(account: OAuthAccountSummary): string {
	const parts = ["oauth"];
	const accountId = account.accountId?.trim();
	if (accountId) parts.push(`account:${accountId}`);
	const email = account.email?.trim().toLowerCase();
	if (email) parts.push(`email:${email}`);
	const orgId = account.orgId?.trim();
	if (orgId) parts.push(`org:${orgId}`);
	const projectId = account.projectId?.trim();
	if (projectId) parts.push(`project:${projectId}`);
	const enterpriseUrl = account.enterpriseUrl?.trim().toLowerCase();
	if (enterpriseUrl) parts.push(`enterprise:${enterpriseUrl}`);
	return parts.join("|");
}

interface UsageIndex {
	/** accountKey -> limitId -> newest recorded row. */
	exact: Map<string, Map<string, UsageHistoryEntry>>;
	/** `account:<id>` / `email:<lower>` -> accountKey, for rows whose key shape we cannot rebuild. */
	aliases: Map<string, string>;
}

function buildUsageIndex(auth: AuthStorage, provider: string, now: number): UsageIndex {
	const index: UsageIndex = { exact: new Map(), aliases: new Map() };
	// Rows arrive oldest-first; a later row for the same window is simply newer.
	for (const entry of auth.usage.history({ provider, sinceMs: now - USAGE_LOOKBACK_MS })) {
		let limits = index.exact.get(entry.accountKey);
		if (!limits) {
			limits = new Map();
			index.exact.set(entry.accountKey, limits);
		}
		const previous = limits.get(entry.limitId);
		if (!previous || previous.recordedAt <= entry.recordedAt) limits.set(entry.limitId, entry);

		const accountId = entry.accountId?.trim().toLowerCase();
		if (accountId) index.aliases.set(`account:${accountId}`, entry.accountKey);
		const email = entry.email?.trim().toLowerCase();
		if (email && !index.aliases.has(`email:${email}`)) index.aliases.set(`email:${email}`, entry.accountKey);
	}
	return index;
}

function lookupUsageRows(index: UsageIndex, account: OAuthAccountSummary): Map<string, UsageHistoryEntry> | undefined {
	const exact = index.exact.get(usageIdentityKey(account));
	if (exact) return exact;
	// Fallback for keys we cannot reconstruct (secret-derived identities, or a
	// credential whose identity fields were filled in after the snapshot).
	// accountId first: it is unique, while an email can repeat across orgs.
	const accountId = account.accountId?.trim().toLowerCase();
	const byAccount = accountId ? index.aliases.get(`account:${accountId}`) : undefined;
	if (byAccount) return index.exact.get(byAccount);
	const email = account.email?.trim().toLowerCase();
	const byEmail = email ? index.aliases.get(`email:${email}`) : undefined;
	return byEmail ? index.exact.get(byEmail) : undefined;
}

/**
 * `anthropic:7d:fable` -> `7d:fable`, `openai-codex:spark:primary` -> `spark:primary`,
 * `google-antigravity:google:default:gemini-weekly` -> `google:gemini-weekly`.
 * Limit ids are the only per-window identifier that stays unique inside one
 * account (`windowLabel` collides: three antigravity meters are all "Weekly").
 */
function shortLimitId(provider: string, limitId: string): string {
	const tail = limitId.startsWith(`${provider}:`) ? limitId.slice(provider.length + 1) : limitId;
	const segments = tail.split(":").filter(segment => segment && segment !== "default");
	if (segments.length === 0) return tail;
	return segments.slice(-2).join(":");
}

function summarizeUsage(
	provider: string,
	rows: Map<string, UsageHistoryEntry> | undefined,
	now: number,
): AccountUsage | undefined {
	if (!rows || rows.size === 0) return undefined;
	const entries = [...rows.values()];
	const recordedAt = Math.max(...entries.map(entry => entry.recordedAt));
	const current = entries.filter(
		(entry): entry is UsageHistoryEntry & { usedFraction: number } =>
			entry.usedFraction !== undefined && entry.recordedAt >= recordedAt - USAGE_SNAPSHOT_SPREAD_MS,
	);
	if (current.length === 0) return undefined;
	current.sort((left, right) => right.usedFraction - left.usedFraction);
	const shown = current.slice(0, USAGE_LIMIT_COLUMNS);
	const resetsAt = shown
		.map(entry => entry.resetsAt)
		.filter((value): value is number => typeof value === "number" && value > now)
		.sort((left, right) => left - right)[0];
	return {
		recordedAt,
		columns: shown.map(entry => `${shortLimitId(provider, entry.limitId)} ${Math.round(entry.usedFraction * 100)}%`),
		resetsAt,
	};
}

function formatDuration(ms: number): string {
	const totalMinutes = Math.max(1, Math.round(ms / 60_000));
	const days = Math.floor(totalMinutes / 1440);
	const hours = Math.floor((totalMinutes % 1440) / 60);
	const minutes = totalMinutes % 60;
	if (days > 0) return hours > 0 ? `${days}d${hours}h` : `${days}d`;
	if (hours > 0) return minutes > 0 ? `${hours}h${minutes}m` : `${hours}h`;
	return `${minutes}m`;
}

/** One line of account detail: usage columns, soonest reset, snapshot age. */
function describeUsage(usage: AccountUsage | undefined, now: number): string {
	if (!usage) return "no usage data";
	const parts = [...usage.columns];
	if (usage.resetsAt) parts.push(`resets in ${formatDuration(usage.resetsAt - now)}`);
	const age = now - usage.recordedAt;
	if (age > USAGE_FRESH_AGE_MS) parts.push(`${formatDuration(age)} ago`);
	return parts.join(" · ");
}

function describeAccount(account: Account, now: number): string {
	const usage = describeUsage(account.usage, now);
	return account.active ? `active · ${usage}` : usage;
}

/**
 * Every provider with at least one stored OAuth account, annotated with the
 * session's active account and each account's last recorded usage.
 */
function collectProviderAccounts(auth: AuthStorage, sessionId: string): ProviderAccounts[] {
	const now = Date.now();
	const providers: ProviderAccounts[] = [];
	for (const provider of Object.keys(auth.credentials.all())) {
		const summaries = auth.oauth.accounts(provider, sessionId);
		if (summaries.length === 0) continue;
		const index = buildUsageIndex(auth, provider, now);
		providers.push({
			provider,
			accounts: summaries.map(summary => ({
				...summary,
				label: formatAccountLabel(summary),
				usage: summarizeUsage(provider, lookupUsageRows(index, summary), now),
			})),
		});
	}
	providers.sort((left, right) => left.provider.localeCompare(right.provider));
	return providers;
}

/**
 * Refresh provider quota reports before reading history. Recording fresh rows
 * is a side effect of the fetch, so this is what keeps the rendered snippets
 * current. Failures (offline, rate limited, no usage provider) are non-fatal:
 * the recorded history is still shown, with its age.
 */
async function refreshUsage(ctx: ExtensionCommandContext): Promise<void> {
	try {
		await ctx.modelRegistry.authStorage.usage.reports({
			baseUrlResolver: provider => ctx.modelRegistry.getProviderBaseUrl(provider),
			signal: AbortSignal.timeout(USAGE_REFRESH_TIMEOUT_MS),
		});
	} catch {
		// Recorded history is the fallback by design.
	}
}

/** Mirrors `/session pin` selector semantics: 1-based position, `active`, or an exact identity field. */
function matchAccounts(accounts: readonly Account[], selector: string): Account[] {
	const wanted = selector.trim().toLowerCase();
	if (!wanted) return [];
	if (wanted === "active") return accounts.filter(account => account.active);
	if (/^\d+$/.test(wanted)) {
		const match = accounts.find(account => account.position === Number(wanted) - 1);
		if (match) return [match];
	}
	return accounts.filter(account =>
		[account.label, account.email, account.accountId, account.projectId, account.enterpriseUrl, account.orgId, account.orgName].some(
			value => value?.trim().toLowerCase() === wanted,
		),
	);
}

/** Single-line feedback. `ui.notify` renders a status line, so it must stay one line. */
function notify(ctx: ExtensionCommandContext, message: string, level: "info" | "warning" | "error" = "info"): void {
	if (ctx.hasUI) ctx.ui.notify(message, level);
	else process.stderr.write(`${message}\n`);
}

/**
 * Multi-line report. Only reachable without a UI (print/headless), where the
 * interactive selector does not exist and stderr is the sane channel — a
 * multi-line `ui.notify` would be squeezed into the TUI status line.
 */
function writeReport(providers: ProviderAccounts[], now: number): void {
	const lines: string[] = [];
	for (const entry of providers) {
		lines.push(entry.provider);
		for (const account of entry.accounts) {
			const marker = account.active ? "*" : " ";
			lines.push(`  ${marker} ${account.position + 1}. ${account.label} — ${describeUsage(account.usage, now)}`);
		}
	}
	lines.push("", "Switch with: /accounts switch <provider> <number|email|active>");
	process.stderr.write(`${lines.join("\n")}\n`);
}

async function pickProvider(
	ctx: ExtensionCommandContext,
	providers: ProviderAccounts[],
): Promise<ProviderAccounts | undefined> {
	// One line per provider: the account-level usage belongs in the next
	// selector, and appending it here wraps the row.
	const options = providers.map(entry => {
		const active = entry.accounts.find(account => account.active);
		return {
			label: entry.provider,
			description: `${entry.accounts.length} account${entry.accounts.length === 1 ? "" : "s"} · ${
				active ? `active: ${active.label}` : "no account pinned yet"
			}`,
		};
	});
	const picked = await ctx.ui.select("Switch account for provider", options, {
		// The provider behind the current model is the likely target.
		initialIndex: Math.max(
			0,
			providers.findIndex(entry => entry.provider === ctx.model?.provider),
		),
	});
	return providers.find(entry => entry.provider === picked);
}

async function pickAccount(
	ctx: ExtensionCommandContext,
	target: ProviderAccounts,
	now: number,
): Promise<Account | undefined> {
	// Positions prefix the label so it stays unique (two credentials can share
	// an email) and so the selector reads like the `<number>` selector argument.
	const options = target.accounts.map(account => ({
		label: `${account.position + 1}. ${account.label}`,
		description: describeAccount(account, now),
	}));
	const picked = await ctx.ui.select(`${target.provider} account`, options, {
		initialIndex: Math.max(
			0,
			target.accounts.findIndex(account => account.active),
		),
	});
	if (!picked) return undefined;
	return target.accounts.find(account => `${account.position + 1}. ${account.label}` === picked);
}

async function showAccountInfo(ctx: ExtensionCommandContext): Promise<void> {
	const providers = collectProviderAccounts(ctx.modelRegistry.authStorage, ctx.sessionManager.getSessionId());
	if (providers.length === 0) {
		notify(ctx, "No stored OAuth accounts. Add one with /login.", "warning");
		return;
	}

	const now = Date.now();
	if (!ctx.hasUI) {
		writeReport(providers, now);
		return;
	}

	// One read-only list spans every provider; selecting a row only closes it.
	await ctx.ui.select(
		"Accounts info (read-only; Enter/Esc to close)",
		providers.flatMap(entry =>
			entry.accounts.map(account => ({
				label: `${entry.provider} · ${account.position + 1}. ${account.label}${account.active ? " (active)" : ""}`,
				description: describeUsage(account.usage, now),
			})),
		),
	);
}

async function runAccountsCommand(args: string, ctx: ExtensionCommandContext): Promise<void> {
	const tokens = args.trim().split(/\s+/).filter(Boolean);
	// Inspect the live session selection without resolving, refreshing, or
	// pinning credentials. Unlike switching, this is safe during a turn.
	if (tokens[0]?.toLowerCase() === "info") {
		if (tokens.length > 1) notify(ctx, "Usage: /accounts info", "warning");
		else await showAccountInfo(ctx);
		return;
	}

	// `/accounts switch anthropic 2` and `/accounts anthropic 2` both work; the
	// verb is optional sugar so the documented spelling and the short one agree.
	if (tokens[0]?.toLowerCase() === "switch" || tokens[0]?.toLowerCase() === "list") tokens.shift();
	const [providerArg, selectorArg] = tokens;

	// Refuse before any prompting: the sticky credential is what the NEXT
	// request reads, so a mid-turn switch would quietly apply to a turn the
	// user is not looking at. Core's `/session pin` refuses for the same
	// reason — and bailing here avoids walking the user through a selector
	// whose result we would then throw away.
	if (!ctx.isIdle()) {
		notify(ctx, "Cannot switch accounts while the agent is streaming. Wait for the turn to finish.", "warning");
		return;
	}

	const auth = ctx.modelRegistry.authStorage;
	// Pick up logins/logouts performed elsewhere since this session started.
	await auth.credentials.reload();
	await refreshUsage(ctx);

	const now = Date.now();
	const sessionId = ctx.sessionManager.getSessionId();
	const providers = collectProviderAccounts(auth, sessionId);
	if (providers.length === 0) {
		notify(ctx, "No stored OAuth accounts. Add one with /login.", "warning");
		return;
	}

	let target = providerArg
		? providers.find(entry => entry.provider.toLowerCase() === providerArg.toLowerCase())
		: undefined;
	if (providerArg && !target) {
		notify(
			ctx,
			`No stored OAuth accounts for "${providerArg}". Providers: ${providers.map(entry => entry.provider).join(", ")}.`,
			"warning",
		);
		return;
	}
	if (!target) {
		if (providers.length === 1) target = providers[0];
		else if (!ctx.hasUI) {
			writeReport(providers, now);
			return;
		} else {
			target = await pickProvider(ctx, providers);
			if (!target) return;
		}
	}
	if (!target) return;

	let chosen: Account | undefined;
	if (selectorArg) {
		const matches = matchAccounts(target.accounts, selectorArg);
		if (matches.length === 0) {
			notify(ctx, `No ${target.provider} account matches "${selectorArg}".`, "warning");
			return;
		}
		if (matches.length > 1) {
			notify(
				ctx,
				`"${selectorArg}" matches ${matches.length} ${target.provider} accounts: ${matches
					.map(account => `${account.position + 1}. ${account.label}`)
					.join(", ")}. Use the number.`,
				"warning",
			);
			return;
		}
		chosen = matches[0];
	} else if (!ctx.hasUI) {
		writeReport([target], now);
		return;
	} else {
		chosen = await pickAccount(ctx, target, now);
		if (!chosen) return;
	}
	if (!chosen) return;

	if (!auth.sessions.pin(target.provider, sessionId, chosen.credentialId)) {
		notify(ctx, `${chosen.label} is no longer available to pin for ${target.provider}.`, "error");
		return;
	}

	const modelProvider = ctx.model?.provider;
	const suffix =
		modelProvider && modelProvider !== target.provider
			? ` (current model runs on ${modelProvider}; takes effect when you switch to a ${target.provider} model)`
			: "";
	notify(ctx, `${target.provider}: this session now uses ${chosen.label}${suffix}.`);
}

export default function accountSwitcher(pi: ExtensionAPI): void {
	pi.setLabel("Account Switcher");

	// Argument autocomplete is synchronous while the provider list comes from
	// the auth store, so it is snapshotted per session binding: on session
	// start and after each run (a `/login` between runs shows up next time).
	let providerCompletions: string[] = [];
	const refreshProviderCompletions = (auth: AuthStorage): void => {
		providerCompletions = Object.keys(auth.credentials.all())
			.filter(provider => auth.oauth.accounts(provider).length > 0)
			.sort();
	};

	pi.registerCommand("accounts", {
		description: "Switch OAuth accounts, or show all accounts with /accounts info",
		getArgumentCompletions: argumentPrefix => {
			const prefix = argumentPrefix.trimStart();
			// Only complete the first argument; past it the selector is an
			// account number/email that autocomplete has nothing to add to.
			if (/\s/.test(prefix)) return null;
			const wanted = prefix.toLowerCase();
			const items = [
				{ value: "info", label: "info", description: "Show all providers' OAuth accounts, active selections, and usage" },
				{ value: "switch", label: "switch", description: "Pick a provider, then an account" },
				...providerCompletions.map(provider => ({
					value: provider,
					label: provider,
					description: "Switch this provider's account",
				})),
			].filter(item => item.value.toLowerCase().startsWith(wanted));
			return items.length > 0 ? items : null;
		},
		handler: async (args, ctx) => {
			try {
				await runAccountsCommand(args, ctx);
			} finally {
				refreshProviderCompletions(ctx.modelRegistry.authStorage);
			}
		},
	});

	pi.on("session_start", (_event, ctx) => {
		refreshProviderCompletions(ctx.modelRegistry.authStorage);
	});
}
