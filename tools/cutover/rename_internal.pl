#!/usr/bin/env perl
# Text rules of the internal stage of rename_to_opensps.sh, applied in place to the files named
# on standard input (one path per line). Idempotent.
#
# All files:
#   fsps_<name>     -> opensps_<name>    (Fortran modules, submodules, procedures, types,
#                                         variables; source file names; Python modules; also
#                                         after an underscore: test_fsps_io -> test_opensps_io)
#   test_fsps       -> test_opensps      (the unit test driver)
#   fsps-test-data  -> opensps-test-data (the in-repository test library)
# Fortran (.f90/.F90) only, outside string literals and comments, not on preprocessor lines:
#   FSPS_<NAME>     -> OPENSPS_<NAME>    (named constants such as FSPS_ERR_UNKNOWN_INT_PARAM),
#                                         except FSPS_VERSION (the FSPS version the physics follows)
# Fortran, also inside strings: the message tags ([FSPS_IO], [FSPS-C], ... -> [OPENSPS_IO],
#   [OPENSPS-C], ...) and the names of the C status codes (FSPS_ERR_*, FSPS_WARN_*, ...).
# Upper-case FSPS_* inside strings (environment variables such as FSPS_DATA_HOME) and on
# preprocessor lines (build switches such as FSPS_PINNED) are left alone.
# meson.build: the unit test names cut the module prefix with substring(5) ('fsps_'); with
#   'opensps_' that is substring(8).
# Fortran lines whose code (without a trailing comment) grows beyond 132 characters are split at
# the last ", " before column 125 that is outside strings, with "&" continuation.
use strict;
use warnings;

my $MAXLEN = 132;

# Split a Fortran line into (code, comment) at the first "!" outside a string
sub split_comment {
    my ($line) = @_;
    my $q = '';
    for (my $i = 0; $i < length($line); $i++) {
        my $c = substr($line, $i, 1);
        if ($q) { $q = '' if $c eq $q; next; }
        if ($c eq "'" || $c eq '"') { $q = $c; next; }
        return (substr($line, 0, $i), substr($line, $i)) if $c eq '!';
    }
    return ($line, '');
}

# Apply $f to the parts of $code outside string literals
sub outside_strings {
    my ($code, $f) = @_;
    my ($out, $buf, $q) = ('', '', '');
    for my $c (split //, $code) {
        if ($q) { $out .= $c; $q = '' if $c eq $q; next; }
        if ($c eq "'" || $c eq '"') { $out .= $f->($buf) . $c; $buf = ''; $q = $c; next; }
        $buf .= $c;
    }
    return $out . $f->($buf);
}

sub split_long {
    my ($line) = @_;
    my ($code, $comment) = split_comment($line);
    (my $bare = $code) =~ s/\s+$//;
    return $line if length($bare) <= $MAXLEN;
    my ($indent) = $code =~ /^(\s*)/;
    # last ", " outside strings before column 125
    my ($q, $cut) = ('', -1);
    for (my $i = 0; $i < length($code) && $i < $MAXLEN - 7; $i++) {
        my $c = substr($code, $i, 1);
        if ($q) { $q = '' if $c eq $q; next; }
        if ($c eq "'" || $c eq '"') { $q = $c; next; }
        $cut = $i if $c eq ',' && substr($code, $i + 1, 1) eq ' ';
    }
    die "rename_internal: cannot split a long line:\n$line\n" if $cut < 0;
    my $first = substr($code, 0, $cut + 1) . ' &';
    my $rest = $indent . '    ' . substr($code, $cut + 2) . $comment;
    return $first . "\n" . split_long($rest);
}

while (my $path = <STDIN>) {
    chomp $path;
    next unless -f $path;
    open(my $in, '<', $path) or die "$path: $!";
    my @lines = <$in>;
    close $in;
    my $fortran = $path =~ /\.[fF]90$/;
    my $changed = 0;
    for my $l (@lines) {
        my $old = $l;
        my $nl = ($l =~ s/\n\z//) ? "\n" : '';
        $l =~ s/(?<![A-Za-z0-9])fsps_(\w)/opensps_$1/g;
        $l =~ s/\btest_fsps\b/test_opensps/g;
        $l =~ s/\bfsps-test-data\b/opensps-test-data/g;
        if ($fortran) {
            $l =~ s/\[FSPS([-_][A-Z]+)\]/[OPENSPS$1]/g;
            $l =~ s/\bFSPS_(ERR|WARN|STATUS|API)_/OPENSPS_$1_/g;
            if ($l !~ /^\s*#/) {
                my ($code, $comment) = split_comment($l);
                $code = outside_strings($code, sub {
                    my $s = shift; $s =~ s/\bFSPS_(?!VERSION\b)([A-Z0-9_]+)\b/OPENSPS_$1/g; $s });
                $l = split_long($code . $comment);
            }
        }
        $l =~ s/unit_module\.substring\(5\)/unit_module.substring(8)/g if $path =~ /(^|\/)meson\.build$/;
        $l .= $nl;
        $changed = 1 if $l ne $old;
    }
    next unless $changed;
    open(my $out, '>', $path) or die "$path: $!";
    print $out @lines;
    close $out;
}
