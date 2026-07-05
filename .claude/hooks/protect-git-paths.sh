#!/usr/bin/env bash
# PreToolUse(Bash) guard: deny *mutating* shell commands that target the
# .git directory or the .gitignore file. Read-only commands that merely
# reference those paths are left alone (the user asked to block mutation
# only). Fails open: if the payload can't be parsed, the command proceeds.
#
# Reads the hook payload as JSON on stdin; on a match it emits a PreToolUse
# "deny" decision and exits 0, otherwise stays silent and exits 0.

command="$(jq -r '.tool_input.command // ""' 2>/dev/null)"

# A .git / .gitignore path token, bounded so that .github and
# .gitattributes do NOT match (only the .git dir and .gitignore file).
path_re='(^|[^[:alnum:]_-])\.git(ignore)?($|[^[:alnum:]_-])'

# Destructive / creating verbs at a command boundary.
verb_re='(^|[;&|(`[:space:]])(rm|mv|cp|touch|mkdir|rmdir|ln|dd|truncate|shred|tee|chmod|chown|chgrp|install)([[:space:]]|$)'

# In-place sed.
sed_re='(^|[[:space:]])sed[[:space:]].*(-i|--in-place)'

# A redirection writing into the protected path (e.g. "> .gitignore").
redir_re='>>?[[:space:]]*\.?/?\.git'

block() {
    printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: mutating commands may not modify .git or .gitignore (protected by the project PreToolUse hook). Run the git operation deliberately or ask the user."}}'
    exit 0
}

# A redirection into .git/.gitignore is self-evidently a write.
if printf '%s' "$command" | grep -Eq "$redir_re"; then
    block
fi

# Otherwise require BOTH a protected path AND a mutating verb / in-place sed.
if printf '%s' "$command" | grep -Eq "$path_re"; then
    if printf '%s' "$command" | grep -Eq "$verb_re" \
        || printf '%s' "$command" | grep -Eq "$sed_re"; then
        block
    fi
fi

exit 0
