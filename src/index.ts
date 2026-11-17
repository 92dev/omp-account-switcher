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
 * The pin holds until the pinned account really fails. `sessions.pin` alone is
 * only a preference: core skips a pinned account that carries a usage block
 * (set by a 429, or by a usage report that reads a window as spent — including
 * a stale block after the quota was reset early), serves from a sibling, and
 * records that sibling as the session's account, dropping the pin for good.
 * So pinning clears the account's blocks and pairs the pin with
 * `sessions.restrict`, which removes every other account from selection,
 * rotation, and fallback. Two rules then run on provider errors
 * ({@link handleProviderError}): an error that leaves a new block on the
 * pinned account lifts the lock so core can rotate through the other
 * accounts, and once every other account is blocked too the session goes back
 * to the pinned account.
 * The pin is recorded in the session file and re-applied on resume;
 * `/accounts unpin` drops it.
 */

import type { SessionRestrictionLease, UsageHistoryEntry } from "@oh-my-pi/pi-ai";
import type {
	AuthStorage,
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
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
	/** This session's pin: `locked` while no other account may serve, `lifted` after the account failed. */
	pin?: "locked" | "lifted";
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

/** `active` is the account serving now; the pin is the account the user chose. */
function describeAccount(account: Account, now: number): string {
	const parts = [describeUsage(account.usage, now)];
	if (account.pin) parts.unshift(account.pin === "locked" ? "pinned" : "pinned, failed over");
	if (account.active) parts.unshift("active");
	return parts.join(" · ");
}

/** Session entry recording a pin; `identityKey: null` records its removal. */
const HARD_PIN_ENTRY = "account-switcher:hard-pin";

interface HardPinEntry {
	provider: string;
	identityKey: string | null;
}

/** The key `sessions.restrict` names an account by; `null` when the credential carries no identity. */
function identityKeyOf(auth: AuthStorage, provider: string, credentialId: number): string | null {
	const entry = auth.credentials
		.snapshot()
		.credentials.find(candidate => candidate.provider === provider && candidate.id === credentialId);
	return entry?.identityKey ?? null;
}

/**
 * This process's pins. A pin names the account the user chose; while `lease`
 * is held the session is restricted to it. Restrictions live in the auth
 * store's memory, so a resumed session gets its pin back from the entries in
 * its session file ({@link restore}).
 */
class HardPins {
	/** `${sessionId}\0${provider}` -> pin. */
	#pins = new Map<string, { identityKey: string; lease?: SessionRestrictionLease }>();

	identityKey(provider: string, sessionId: string): string | undefined {
		return this.#pins.get(`${sessionId}\0${provider}`)?.identityKey;
	}

	isLocked(provider: string, sessionId: string): boolean {
		return this.#pins.get(`${sessionId}\0${provider}`)?.lease !== undefined;
	}

	providers(sessionId: string): string[] {
		const prefix = `${sessionId}\0`;
		return [...this.#pins.keys()].filter(key => key.startsWith(prefix)).map(key => key.slice(prefix.length));
	}

	/**
	 * Restrict the session to the account and clear the account's blocks, so
	 * that a block found on it later was set by a failure after this call.
	 */
	lock(auth: AuthStorage, provider: string, sessionId: string, identityKey: string): void {
		const target = auth.credentials
			.snapshot()
			.credentials.find(entry => entry.provider === provider && entry.identityKey === identityKey);
		if (target) {
			for (const block of auth.blocks.list([target.id])) {
				auth.blocks.delete(block.credentialId, block.providerKey, block.blockScope);
			}
		}
		// `restrict` replaces an earlier restriction for the same provider and session.
		const lease = auth.sessions.restrict(provider, sessionId, [identityKey]);
		this.#pins.set(`${sessionId}\0${provider}`, { identityKey, lease });
	}

	/** Lift the restriction; with `keepPin` the session still remembers the account. */
	unlock(auth: AuthStorage, provider: string, sessionId: string, keepPin = false): void {
		const key = `${sessionId}\0${provider}`;
		const pin = this.#pins.get(key);
		if (!pin) return;
		if (pin.lease) auth.sessions.unrestrict(provider, sessionId, pin.lease);
		if (keepPin) this.#pins.set(key, { identityKey: pin.identityKey });
		else this.#pins.delete(key);
	}

	/** Re-apply the pins recorded in the session file. The last entry per provider wins. */
	restore(ctx: ExtensionContext): void {
		const auth = ctx.modelRegistry.authStorage;
		const sessionId = ctx.sessionManager.getSessionId();
		const recorded = new Map<string, string | null>();
		for (const entry of ctx.sessionManager.getEntries()) {
			if (entry.type !== "custom" || entry.customType !== HARD_PIN_ENTRY) continue;
			const data = entry.data as HardPinEntry | undefined;
			if (data?.provider) recorded.set(data.provider, data.identityKey ?? null);
		}
		if (recorded.size === 0) return;
		const stored = auth.credentials.snapshot().credentials;
		for (const [provider, identityKey] of recorded) {
			if (identityKey === null || this.identityKey(provider, sessionId) === identityKey) continue;
			// Restricting to a logged-out account would leave the provider with no usable account.
			if (!stored.some(entry => entry.provider === provider && entry.identityKey === identityKey)) {
				if (ctx.hasUI) {
					ctx.ui.notify(`${provider}: pinned account ${identityKey} is no longer stored; pin dropped.`, "warning");
				}
				continue;
			}
			this.lock(auth, provider, sessionId, identityKey);
		}
	}
}

/**
 * Keep a pin honest after a provider error on the current model's provider.
 *
 * Locked: core blocks an account on a usage limit, policy denial, or auth
 * failure, and {@link HardPins.lock} cleared older blocks, so a block on the
 * pinned account means it failed since. Lift the lock and let core fail over.
 * A transient error leaves no block and the lock stays.
 *
 * Lifted: core rotates through the other accounts on its own, blocking each
 * one that fails. Once all of them are blocked nothing is left to try; go back
 * to the account the user chose.
 */
function handleProviderError(ctx: ExtensionContext, hardPins: HardPins): void {
	const provider = ctx.model?.provider;
	if (!provider) return;
	const auth = ctx.modelRegistry.authStorage;
	const sessionId = ctx.sessionManager.getSessionId();
	const identityKey = hardPins.identityKey(provider, sessionId);
	if (identityKey === undefined) return;
	const pinned = auth.credentials
		.snapshot()
		.credentials.find(entry => entry.provider === provider && entry.identityKey === identityKey);
	if (!pinned) return;

	if (hardPins.isLocked(provider, sessionId)) {
		if (auth.blocks.list([pinned.id]).length === 0) return;
		hardPins.unlock(auth, provider, sessionId, true);
		if (ctx.hasUI) {
			ctx.ui.notify(`${provider}: pinned account hit a limit or auth error; omp may fail over to another account.`, "warning");
		}
		return;
	}
	const others = auth.oauth
		.accounts(provider, sessionId)
		.map(account => account.credentialId)
		.filter(credentialId => credentialId !== pinned.id);
	const blocked = new Set(auth.blocks.list(others).map(block => block.credentialId));
	if (others.some(credentialId => !blocked.has(credentialId))) return;
	hardPins.lock(auth, provider, sessionId, identityKey);
	auth.sessions.pin(provider, sessionId, pinned.id);
	if (ctx.hasUI) ctx.ui.notify(`${provider}: every other account failed too; back on the pinned account.`, "warning");
}

/**
 * A lifted pin whose account served a turn again has recovered: lock it, which
 * also clears the block its failure left, so that block cannot move the
 * session once the other accounts' blocks expire.
 */
function handleProviderSuccess(ctx: ExtensionContext, hardPins: HardPins): void {
	const provider = ctx.model?.provider;
	if (!provider) return;
	const auth = ctx.modelRegistry.authStorage;
	const sessionId = ctx.sessionManager.getSessionId();
	const identityKey = hardPins.identityKey(provider, sessionId);
	if (identityKey === undefined || hardPins.isLocked(provider, sessionId)) return;
	const active = auth.oauth.accounts(provider, sessionId).find(account => account.active);
	if (!active || identityKeyOf(auth, provider, active.credentialId) !== identityKey) return;
	hardPins.lock(auth, provider, sessionId, identityKey);
}

/**
 * Every provider with at least one stored OAuth account, annotated with the
 * session's active account and each account's last recorded usage.
 */
function collectProviderAccounts(auth: AuthStorage, sessionId: string, hardPins: HardPins): ProviderAccounts[] {
	const identityKeys = new Map(auth.credentials.snapshot().credentials.map(entry => [entry.id, entry.identityKey]));
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
				...(hardPins.identityKey(provider, sessionId) !== undefined &&
				hardPins.identityKey(provider, sessionId) === identityKeys.get(summary.credentialId)
					? { pin: hardPins.isLocked(provider, sessionId) ? ("locked" as const) : ("lifted" as const) }
					: {}),
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
			const marks = account.pin ? ` (${account.pin === "locked" ? "pinned" : "pinned, failed over"})` : "";
			lines.push(`  ${marker} ${account.position + 1}. ${account.label}${marks} — ${describeUsage(account.usage, now)}`);
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

async function showAccountInfo(ctx: ExtensionCommandContext, hardPins: HardPins): Promise<void> {
	const providers = collectProviderAccounts(
		ctx.modelRegistry.authStorage,
		ctx.sessionManager.getSessionId(),
		hardPins,
	);
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
				label: `${entry.provider} · ${account.position + 1}. ${account.label}`,
				description: describeAccount(account, now),
			})),
		),
	);
}

/** `/accounts unpin [provider]`: drop this session's pin for one provider, or for all of them. */
function unpinAccounts(pi: ExtensionAPI, ctx: ExtensionCommandContext, hardPins: HardPins, providerArg?: string): void {
	const auth = ctx.modelRegistry.authStorage;
	const sessionId = ctx.sessionManager.getSessionId();
	const pinned = hardPins.providers(sessionId);
	const targets = providerArg ? pinned.filter(provider => provider.toLowerCase() === providerArg.toLowerCase()) : pinned;
	if (targets.length === 0) {
		notify(ctx, providerArg ? `No pinned account for "${providerArg}" in this session.` : "No pinned accounts in this session.", "warning");
		return;
	}
	for (const provider of targets) {
		hardPins.unlock(auth, provider, sessionId);
		pi.appendEntry<HardPinEntry>(HARD_PIN_ENTRY, { provider, identityKey: null });
	}
	notify(ctx, `${targets.join(", ")}: unpinned; omp may switch accounts again.`);
}

async function runAccountsCommand(
	args: string,
	ctx: ExtensionCommandContext,
	pi: ExtensionAPI,
	hardPins: HardPins,
): Promise<void> {
	const tokens = args.trim().split(/\s+/).filter(Boolean);
	// Inspect the live session selection without resolving, refreshing, or
	// pinning credentials. Unlike switching, this is safe during a turn.
	if (tokens[0]?.toLowerCase() === "info") {
		if (tokens.length > 1) notify(ctx, "Usage: /accounts info", "warning");
		else await showAccountInfo(ctx, hardPins);
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

	if (tokens[0]?.toLowerCase() === "unpin") {
		if (tokens.length > 2) notify(ctx, "Usage: /accounts unpin [provider]", "warning");
		else unpinAccounts(pi, ctx, hardPins, tokens[1]);
		return;
	}

	const auth = ctx.modelRegistry.authStorage;
	// Pick up logins/logouts performed elsewhere since this session started.
	await auth.credentials.reload();
	await refreshUsage(ctx);

	const now = Date.now();
	const sessionId = ctx.sessionManager.getSessionId();
	const providers = collectProviderAccounts(auth, sessionId, hardPins);
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

	// Lock before pinning: a session locked to another account refuses a pin
	// outside its restriction.
	const identityKey = identityKeyOf(auth, target.provider, chosen.credentialId);
	const previousKey = hardPins.identityKey(target.provider, sessionId);
	if (identityKey !== null) hardPins.lock(auth, target.provider, sessionId, identityKey);
	else hardPins.unlock(auth, target.provider, sessionId);
	if (!auth.sessions.pin(target.provider, sessionId, chosen.credentialId)) {
		if (previousKey !== undefined) hardPins.lock(auth, target.provider, sessionId, previousKey);
		else hardPins.unlock(auth, target.provider, sessionId);
		notify(ctx, `${chosen.label} is no longer available to pin for ${target.provider}.`, "error");
		return;
	}
	if (identityKey !== previousKey) {
		pi.appendEntry<HardPinEntry>(HARD_PIN_ENTRY, { provider: target.provider, identityKey });
	}

	const modelProvider = ctx.model?.provider;
	const suffix =
		modelProvider && modelProvider !== target.provider
			? ` (current model runs on ${modelProvider}; takes effect when you switch to a ${target.provider} model)`
			: "";
	if (identityKey === null) {
		notify(
			ctx,
			`${target.provider}: pinned ${chosen.label}, but it has no account identity to lock on; omp may still switch away${suffix}.`,
			"warning",
		);
		return;
	}
	notify(
		ctx,
		`${target.provider}: pinned ${chosen.label}; omp stays on it until it fails${suffix}. /accounts unpin drops the pin.`,
	);
}

export default function accountSwitcher(pi: ExtensionAPI): void {
	pi.setLabel("Account Switcher");

	// Argument autocomplete is synchronous while the provider list comes from
	// the auth store, so it is snapshotted per session binding: on session
	// start and after each run (a `/login` between runs shows up next time).
	const hardPins = new HardPins();
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
				{ value: "switch", label: "switch", description: "Pick a provider, then an account, and lock the session to it" },
				{ value: "unpin", label: "unpin", description: "Lift this session's account lock" },
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
				await runAccountsCommand(args, ctx, pi, hardPins);
			} finally {
				refreshProviderCompletions(ctx.modelRegistry.authStorage);
			}
		},
	});

	pi.on("session_start", (_event, ctx) => {
		refreshProviderCompletions(ctx.modelRegistry.authStorage);
		hardPins.restore(ctx);
	});
	// A switch, fork, or branch can land on another session id, whose pins are its own.
	pi.on("session_switch", (_event, ctx) => hardPins.restore(ctx));
	pi.on("session_branch", (_event, ctx) => hardPins.restore(ctx));
	// A failed attempt surfaces as a retry announcement, a failed retry saga, or an errored turn.
	pi.on("auto_retry_start", (_event, ctx) => handleProviderError(ctx, hardPins));
	pi.on("auto_retry_end", (event, ctx) => {
		if (!event.success) handleProviderError(ctx, hardPins);
	});
	pi.on("turn_end", (event, ctx) => {
		if (event.message.role !== "assistant") return;
		if (event.message.stopReason === "error") handleProviderError(ctx, hardPins);
		else if (event.message.stopReason !== "aborted") handleProviderSuccess(ctx, hardPins);
	});
}
