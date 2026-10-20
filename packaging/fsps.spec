Name:           fsps
Version:        3.2.0
Release:        1%{?dist}
Summary:        Flexible Stellar Population Synthesis (FSPS)

License:        MIT
URL:            https://github.com/elijahmathews/fsps
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc-gfortran
BuildRequires:  meson >= 0.57.0
BuildRequires:  ninja-build

# The base 'fsps' package is empty in favor of explicit subpackages.

%description
Flexible Stellar Population Synthesis (FSPS) is a stellar population
synthesis code.

# ------------------------------------------------------------------
# 1. Shared Library Package
# ------------------------------------------------------------------
%package -n libfsps
Summary:        FSPS shared library
Requires:       libgfortran

%description -n libfsps
Shared library for FSPS. This is the runtime dependency needed by 
language wrappers and external applications.

# ------------------------------------------------------------------
# 2. Development Package
# ------------------------------------------------------------------
%package -n libfsps-devel
Summary:        Development files for libfsps
Requires:       libfsps%{?_isa} = %{version}-%{release}
Provides:       pkgconfig(fsps)

%description -n libfsps-devel
Header files, unversioned linker symlink, and pkg-config metadata 
for developing applications against libfsps.

# ------------------------------------------------------------------
# Build & Install Stages
# ------------------------------------------------------------------
%prep
# Unpacks the Source0 tarball automatically
%autosetup -n %{name}-%{version}

%build
# The meson macro automatically sets --prefix=/usr, --libdir=/usr/lib64, etc.
# We explicitly pass our new optimization toggle to guarantee a generic, 
# highly-performant binary for distribution.
%meson -Dcpu_baseline=x86-64-v3
%meson_build

%install
%meson_install

# Since we no longer package the data sets, we must delete them from the 
# buildroot so rpmbuild does not throw an "unpackaged files" fatal error.
rm -rf %{buildroot}%{_datadir}/fsps

%check
# Meson inherently handles FSPS_DATA_HOME via the regression_env object 
# defined in meson.build, so we can just run the native macro.
%meson_test

# ------------------------------------------------------------------
# File Manifests
# ------------------------------------------------------------------
%files -n libfsps
%license LICENSE.md
# Package the versioned shared objects
%{_libdir}/libfsps.so.3*

%files -n libfsps-devel
%{_includedir}/fsps.h
# Package the unversioned symlink used by the linker
%{_libdir}/libfsps.so
%{_libdir}/pkgconfig/fsps.pc

%changelog
* Thu Jan 29 2026 FSPS Packager <23229627+elijahmathews@users.noreply.github.com> - 3.2.0-1
- Transitioned to Meson build system
- Dropped CLI executables and external data payload
- Split into libfsps and libfsps-devel