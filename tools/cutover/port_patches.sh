#!/usr/bin/env bash
# Port commits of elijahmathews/fsps (as git format-patch files) onto renamed libopensps.
#
#   tools/cutover/port_patches.sh [options] <patch>...
#
# Run inside a libopensps clone, on the branch that should receive the commits (e.g. a topic
# branch from main). A fork patch names fsps paths and identifiers, so it cannot be applied with
# git am directly. For each patch, in order, the script:
#
#   1. applies it with git am on top of the import commit (where the tree is the fork's), in a
#      scratch worktree, together with the patches before it;
#   2. runs rename_to_opensps.sh (the same rename as main's) on that tree and on the tree before
#      the patch;
#   3. applies the difference between the two renamed trees to the current branch (git apply -3)
#      and commits it with the patch's author, date and message.
#
# So the patch arrives renamed exactly as the code it touches was. The deprecated fsps.h is left
# out of the difference (it is frozen at the rename). Conflicts stop the script with the
# three-way merge state in the working tree, like git am. It never pushes.
#
# Options:
#   --import REV     the import commit (default: the newest commit on origin/main whose subject
#                    starts with "feat: import elijahmathews/fsps")
#   --rename-args A  extra arguments for rename_to_opensps.sh (default: none, i.e. its defaults,
#                    which main's rename used)
#   --prefix TEXT    prepend TEXT to each commit subject (e.g. "fix: "; libopensps uses
#                    Conventional Commits; default: keep the subject and add nothing)
#   -h, --help       this text
set -euo pipefail

die() { echo "port_patches: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed -e '/^set -euo/d' -e 's/^# \{0,1\}//'; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
import=""; rename_args=""; prefix=""
patches=()
while [ $# -gt 0 ]; do
    case "$1" in
        --import) import=$2; shift 2 ;;
        --rename-args) rename_args=$2; shift 2 ;;
        --prefix) prefix=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option $1 (see --help)" ;;
        *) patches+=("$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"); shift ;;
    esac
done
[ ${#patches[@]} -gt 0 ] || { usage >&2; exit 2; }
for p in "${patches[@]}"; do [ -f "$p" ] || die "no patch $p"; done

top=$(git rev-parse --show-toplevel) || die "run inside a libopensps clone"
cd "$top"
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "the working tree has changes"
if [ -z "$import" ]; then
    import=$(git log --format=%H -1 --grep='^feat: import elijahmathews/fsps' origin/main) || true
    [ -n "$import" ] || die "no import commit found on origin/main; pass --import"
fi
import=$(git rev-parse --verify "${import}^{commit}")

tmp=$(mktemp -d)
cleanup() {
    git worktree remove --force "$tmp/plain" 2>/dev/null || true
    git worktree remove --force "$tmp/renamed" 2>/dev/null || true
    rm -rf "$tmp"
}
trap cleanup EXIT

# The renamed tree of a commit, as a commit object (in a scratch worktree, detached)
renamed_of() {
    git worktree add -q --detach "$tmp/renamed" "$1"
    (
        cd "$tmp/renamed"
        # shellcheck disable=SC2086
        "$script_dir/rename_to_opensps.sh" $rename_args . > "$tmp/rename.log" 2>&1 ||
            { cat "$tmp/rename.log" >&2; exit 1; }
        git add -A
        git -c user.name=port -c user.email=port@invalid -c commit.gpgsign=false \
            commit -q --no-verify -m "renamed $1"
        git rev-parse HEAD
    )
    git worktree remove --force "$tmp/renamed"
}

git worktree add -q --detach "$tmp/plain" "$import"
before=$(renamed_of "$import")
for p in "${patches[@]}"; do
    echo "== $(basename "$p")"
    git -C "$tmp/plain" -c commit.gpgsign=false am -q --3way "$p" ||
        die "$(basename "$p") does not apply to the import commit (resolve in $tmp/plain, or port it by hand)"
    plain=$(git -C "$tmp/plain" rev-parse HEAD)
    after=$(renamed_of "$plain")
    git diff --binary "$before" "$after" -- . ':(exclude)include/fsps.h' > "$tmp/port.diff"
    if [ ! -s "$tmp/port.diff" ]; then
        echo "  nothing to apply (already in the tree?)"
    else
        git apply --3way --index "$tmp/port.diff" ||
            die "conflicts porting $(basename "$p"): resolve, then commit with: git commit -c $plain"
        orig=$(sed -n '1s/^From \([0-9a-f]\{40\}\) .*/\1/p' "$p")
        subject=$(git log -1 --format=%s "$plain")
        body=$(git log -1 --format=%b "$plain")
        GIT_AUTHOR_NAME=$(git log -1 --format=%an "$plain") GIT_AUTHOR_EMAIL=$(git log -1 --format=%ae "$plain") \
        GIT_AUTHOR_DATE=$(git log -1 --format=%aD "$plain") \
            git -c commit.gpgsign=false commit -q --no-verify -F - <<EOF
$prefix$subject

$body${orig:+
Ported-from: elijahmathews/fsps@$orig}
EOF
        echo "  $(git log -1 --format='%h %s')"
    fi
    before=$after
done
echo "Ported ${#patches[@]} patch(es) onto $(git rev-parse --abbrev-ref HEAD). Nothing was pushed."
