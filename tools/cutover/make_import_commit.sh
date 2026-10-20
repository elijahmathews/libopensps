#!/usr/bin/env bash
# Import one commit of elijahmathews/fsps into a libopensps checkout, as ONE commit on top of its
# current main.
#
#   tools/cutover/make_import_commit.sh [options] <fork-commit> <libopensps-dir>
#
# <libopensps-dir> is a clone of elijahmathews/libopensps. The script creates a branch (default
# `import`) from the base (default origin/main), lays the fork's tree at <fork-commit> on top of
# it and commits once:
#
#   feat: import elijahmathews/fsps at commit <full hash>
#
# with a body naming the source, what was left out and where the history and authorship are
# recorded. It never pushes and never adds a remote. What is imported:
#
#   left out   data/ (the libraries, published as data releases of elijahmathews/opensps-data);
#              the regression references tests/data/sps_ref_*.bin and tests/data/param_ref/
#              (part of the data release; copied with SHA256SUMS to --refs-dir for checking)
#   kept       every file the base already has: the repository's own README.md, LICENSE.md,
#              AGENTS.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, assets/ (and anything
#              else on the base); the fork's versions of such files are not imported (the script
#              lists them; tools/cutover/proposals/ has what is worth merging into them)
#   added      AUTHORS (from the fork's history, identities merged by authors.mailmap) and
#              CITATION.cff (tools/cutover/templates/CITATION.cff), unless the base has them
#   adjusted   CI branch filters of the imported workflows ([master, develop] -> [main]) so that
#              CI runs on the import pull request (--ci-branch '' leaves them as they are)
#
# Options:
#   --repo DIR            the fork's clone (default: the repository holding this script)
#   --base REV            base of the import in <libopensps-dir> (default: origin/main)
#   --branch NAME         branch to create (default: import)
#   --author "N <e>"      author and committer (default: git config user.* of <libopensps-dir>)
#   --fsps-base REV       last upstream FSPS commit in the fork's history; AUTHORS credits commits
#                         up to it to FSPS, later ones to the fork (default: merge-base with
#                         upstream/master if that remote exists, else 82a873508d50, the fork point)
#   --refs-dir DIR        where the left-out references are copied, with SHA256SUMS
#                         (default: <libopensps-dir>.references)
#   --keep-refs           import the regression references too
#   --ci-branch NAME      branch for the imported CI filters (default: main; '' = unchanged)
#   --trailer TEXT        append a trailer to the commit message (repeatable; none by default)
#   --date DATE           author/committer date (default: now)
#   -h, --help            this text
set -euo pipefail

die() { echo "make_import_commit: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed -e '/^set -euo/d' -e 's/^# \{0,1\}//'; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=""; base="origin/main"; branch="import"; author=""; fsps_base=""; refs_dir=""; keep_refs=0
ci_branch="main"; date=""; trailers=()
default_fsps_base=82a873508d500ca353bbb922459bf928498f7a72

args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) repo=$2; shift 2 ;;
        --base) base=$2; shift 2 ;;
        --branch) branch=$2; shift 2 ;;
        --author) author=$2; shift 2 ;;
        --fsps-base) fsps_base=$2; shift 2 ;;
        --refs-dir) refs_dir=$2; shift 2 ;;
        --keep-refs) keep_refs=1; shift ;;
        --ci-branch) ci_branch=$2; shift 2 ;;
        --trailer) trailers+=("$2"); shift 2 ;;
        --date) date=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option $1 (see --help)" ;;
        *) args+=("$1"); shift ;;
    esac
done
[ ${#args[@]} -eq 2 ] || { usage >&2; exit 2; }
rev=${args[0]}
dest=$(cd "${args[1]}" 2>/dev/null && pwd) || die "${args[1]} does not exist (clone libopensps there first)"

[ -n "$repo" ] || repo=$(git -C "$script_dir" rev-parse --show-toplevel)
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || die "$repo is not a git repository"
commit=$(git -C "$repo" rev-parse --verify "${rev}^{commit}") || die "no commit $rev in $repo"
git -C "$dest" rev-parse --git-dir >/dev/null 2>&1 || die "$dest is not a git repository"
[ -z "$(git -C "$dest" status --porcelain)" ] || die "$dest has uncommitted changes"
base_commit=$(git -C "$dest" rev-parse --verify "${base}^{commit}") || die "no base $base in $dest (git fetch?)"
git -C "$dest" rev-parse --verify -q "refs/heads/$branch" >/dev/null && die "branch $branch exists in $dest"

if [ -z "$author" ]; then
    name=$(git -C "$dest" config user.name) || die "no user.name in $dest; pass --author"
    email=$(git -C "$dest" config user.email) || die "no user.email in $dest; pass --author"
    author="$name <$email>"
fi
[[ "$author" =~ ^(.+)\ \<([^>]+)\>$ ]] || die "--author must be 'Name <email>'"
author_name=${BASH_REMATCH[1]}
author_email=${BASH_REMATCH[2]}

if [ -z "$fsps_base" ]; then
    if git -C "$repo" rev-parse --verify -q upstream/master >/dev/null; then
        fsps_base=$(git -C "$repo" merge-base "$commit" upstream/master)
    else
        fsps_base=$default_fsps_base
    fi
fi
fsps_base=$(git -C "$repo" rev-parse --verify "${fsps_base}^{commit}") || die "no commit $fsps_base"
git -C "$repo" merge-base --is-ancestor "$fsps_base" "$commit" || die "$fsps_base is not an ancestor of $commit"

[ -n "$refs_dir" ] || refs_dir="$dest.references"
if [ $keep_refs -eq 0 ] && [ -n "$(ls -A "$refs_dir" 2>/dev/null)" ]; then
    die "$refs_dir exists and is not empty (pass --refs-dir)"
fi

# ---------------------------------------------------------------------------------------------
# The fork's tree, in a staging directory
# ---------------------------------------------------------------------------------------------
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
echo "== Exporting $commit"
git -C "$repo" archive --format=tar "$commit" | tar -x -C "$stage"

removed=()
if [ -d "$stage/data" ]; then
    removed+=("data/ ($(du -sh "$stage/data" | cut -f1)B), the libraries: published as data releases of elijahmathews/opensps-data")
    rm -rf "$stage/data"
fi
if [ $keep_refs -eq 0 ]; then
    shopt -s nullglob
    refs=("$stage"/tests/data/sps_ref_*.bin)
    shopt -u nullglob
    [ -d "$stage/tests/data/param_ref" ] && refs+=("$stage/tests/data/param_ref")
    if [ ${#refs[@]} -gt 0 ]; then
        mkdir -p "$refs_dir"
        refs_dir=$(cd "$refs_dir" && pwd)
        refs_size=$(du -csh "${refs[@]}" | tail -1 | cut -f1)
        cp -r "${refs[@]}" "$refs_dir/"
        (cd "$refs_dir" && find . -type f ! -name SHA256SUMS -printf '%P\n' | sort | xargs sha256sum > SHA256SUMS)
        removed+=("tests/data/sps_ref_*.bin and tests/data/param_ref/ (${refs_size}B), the regression references: part of the data release")
        rm -rf "${refs[@]}"
    fi
fi

# The base's files win: the fork's versions of files that already exist are not imported
git -C "$dest" ls-tree -r --name-only "$base_commit" > "$stage.base-files"
kept=()
while IFS= read -r f; do
    if [ -e "$stage/$f" ]; then
        kept+=("$f")
        rm -f "$stage/$f"
    fi
done < "$stage.base-files"
rm -f "$stage.base-files"

# CI filters: run on the import pull request
if [ -n "$ci_branch" ] && [ -d "$stage/.github/workflows" ]; then
    for wf in "$stage"/.github/workflows/*.yml; do
        perl -i -pe "s/^(\\s*branches:\\s*)\\[(?:master, develop|master|develop)\\]/\${1}[$ci_branch]/" "$wf"
    done
fi

echo "== Files above 1 MB in the import"
big=$(cd "$stage" && find . -type f -size +1024k -printf '%s\t%P\n' | sort -rn)
if [ -n "$big" ]; then
    printf '%s\n' "$big" | awk -F'\t' '{printf "  %6.1f MB  %s\n", $1/1048576, $2}'
else
    echo "  (none)"
fi

# ---------------------------------------------------------------------------------------------
# AUTHORS (from the fork's history) and CITATION.cff
# ---------------------------------------------------------------------------------------------
mailmap="$script_dir/authors.mailmap"
[ -f "$mailmap" ] || die "missing $mailmap"

# People ("Name <email>" after the mailmap) by number of commits over a revision range; bots and
# AI agents are left out (noted separately)
people() {
    git -C "$repo" -c mailmap.file="$mailmap" log --use-mailmap --format='%aN <%aE>' "$@" |
        grep -v -E '^(Claude|Google Code Exporter|GitHub) <' |
        sort | uniq -c | sort -k1,1nr -k2 | sed 's/^ *[0-9]* //'
}
format_people() {
    local named="" handles=""
    while IFS= read -r who; do
        local nm=${who% <*}
        if [[ "$nm" == *@* ]]; then
            handles+="${handles:+, }${nm%%@*}"
        else
            named+="- $nm"$'\n'
        fi
    done
    printf '%s' "$named"
    [ -n "$handles" ] && printf '\nContributors from the Google Code era of FSPS, known only by these handles: %s.\n' "$handles"
    return 0
}
fsps_people=$(people "$fsps_base" | format_people)
fork_people=$(people "$fsps_base..$commit" | format_people)
n_ai=$(git -C "$repo" log --format='%aE%n%(trailers:key=Co-authored-by,valueonly)' "$fsps_base..$commit" |
       grep -c -i 'noreply@anthropic.com' || true)
short_base=$(git -C "$repo" rev-parse --short=12 "$fsps_base")

if ! git -C "$dest" cat-file -e "$base_commit:AUTHORS" 2>/dev/null; then
    cat > "$stage/AUTHORS" <<EOF
# Authors

libopensps is derived from FSPS, Flexible Stellar Population Synthesis, the work of Charlie
Conroy and contributors (https://github.com/cconroy20/fsps). The models are described in
Conroy, Gunn & White (2009, ApJ, 699, 486) and Conroy & Gunn (2010, ApJ, 712, 833); please cite
both papers in work that uses this library (CITATION.cff).

The code was imported from the modernized fork elijahmathews/fsps at commit
$commit
without its history. The full history, which records who wrote what, stays available in that
repository (tag fsps-final). The lists below are derived from it, ordered by number of commits.

## FSPS (commit history up to $short_base, where the fork began)

$fsps_people

## The modernized fork (elijahmathews/fsps)

$fork_people
EOF
    if [ "$n_ai" -gt 0 ]; then
        cat >> "$stage/AUTHORS" <<'EOF'

Much of the fork's code was written with AI coding agents (Claude, by Anthropic) under the
direction and review of the fork's maintainer; the commit messages record them as co-authors.
EOF
    fi
fi
if ! git -C "$dest" cat-file -e "$base_commit:CITATION.cff" 2>/dev/null && [ -f "$script_dir/templates/CITATION.cff" ]; then
    cp "$script_dir/templates/CITATION.cff" "$stage/CITATION.cff"
fi

# ---------------------------------------------------------------------------------------------
# The commit
# ---------------------------------------------------------------------------------------------
echo "== Committing on branch $branch (from $base, $(git -C "$dest" rev-parse --short "$base_commit"))"
git -C "$dest" switch -q -c "$branch" "$base_commit"
(cd "$stage" && tar -cf - .) | tar -xf - -C "$dest"
git -C "$dest" add -A

bullet() { printf '%s\n' "$1" | fold -s -w 70 | sed -e '1s/^/- /' -e '2,$s/^/  /' -e 's/ *$//'; }
body="Source: https://github.com/elijahmathews/fsps (tag fsps-final), commit
$commit,
the modernized FSPS fork. Its tree is imported without its history, except:"$'\n'
for r in "${removed[@]}"; do body+=$(bullet "left out: $r;")$'\n'; done
if [ ${#kept[@]} -gt 0 ]; then
    body+=$(bullet "not imported, this repository's own files kept instead: $(printf '%s, ' "${kept[@]}" | sed 's/, $//');")$'\n'
fi
[ -n "$ci_branch" ] && body+=$(bullet "CI workflows run on $ci_branch (they ran on master and develop);")$'\n'
body+=$(bullet "added: AUTHORS, CITATION.cff.")
msg="feat: import elijahmathews/fsps at commit $commit

$body

History and authorship are recorded in AUTHORS, derived from the fork's
history; the full history stays in elijahmathews/fsps."
if [ ${#trailers[@]} -gt 0 ]; then
    msg+=$'\n'
    for t in "${trailers[@]}"; do msg+=$'\n'"$t"; done
fi
date_env=()
[ -n "$date" ] && date_env=(GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date")
env "${date_env[@]}" GIT_AUTHOR_NAME="$author_name" GIT_AUTHOR_EMAIL="$author_email" \
    GIT_COMMITTER_NAME="$author_name" GIT_COMMITTER_EMAIL="$author_email" \
    git -C "$dest" -c commit.gpgsign=false commit -q --no-verify -m "$msg"

echo
echo "== Done: branch $branch in $dest"
git -C "$dest" log --format='  %H%n  %an <%ae>%n  %s' -1
echo "  $(git -C "$dest" diff --shortstat "$base_commit" HEAD)"
for r in "${removed[@]}"; do echo "  left out: $r"; done
[ $keep_refs -eq 1 ] || echo "  references copied to $refs_dir"
[ ${#kept[@]} -eq 0 ] || echo "  kept from the base (fork's version not imported): ${kept[*]}"
echo "No remote was changed and nothing was pushed."
