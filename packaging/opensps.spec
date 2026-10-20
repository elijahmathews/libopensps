Name:           opensps
Version:        0.1.0
Release:        1%{?dist}
Summary:        Stellar population synthesis library derived from FSPS

License:        MIT
URL:            https://github.com/elijahmathews/libopensps
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc-gfortran
BuildRequires:  meson >= 0.57.0
BuildRequires:  ninja-build

# The base 'opensps' package is empty in favor of explicit subpackages.

%description
libopensps is a stellar population synthesis library derived from FSPS
(Flexible Stellar Population Synthesis, Conroy et al.).

# ------------------------------------------------------------------
# 1. Shared Library Package
# ------------------------------------------------------------------
%package -n libopensps
Summary:        FSPS shared library
Requires:       libgfortran

%description -n libopensps
Shared library for FSPS. This is the runtime dependency needed by 
language wrappers and external applications.

# ------------------------------------------------------------------
# 2. Development Package
# ------------------------------------------------------------------
%package -n libopensps-devel
Summary:        Development files for libopensps
Requires:       libopensps%{?_isa} = %{version}-%{release}
Provides:       pkgconfig(opensps)

%description -n libopensps-devel
Header files, unversioned linker symlink, and pkg-config metadata 
for developing applications against libopensps.

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
# The package is built without the data (a separate release, doc/DATA_FORMAT.md):
# run the unit suite, which needs neither data nor network.
%meson_test --suite unit

# ------------------------------------------------------------------
# File Manifests
# ------------------------------------------------------------------
%files -n libopensps
%license LICENSE.md
# Package the versioned shared objects
%{_libdir}/libopensps.so.0*

%files -n libopensps-devel
%{_includedir}/opensps.h
%{_includedir}/fsps.h
# Package the unversioned symlink used by the linker
%{_libdir}/libopensps.so
%{_libdir}/pkgconfig/opensps.pc

%changelog
* Thu Jan 29 2026 FSPS Packager <23229627+elijahmathews@users.noreply.github.com> - 0.1.0-1
- Renamed from fsps to opensps (libopensps 0.1.0, soname libopensps.so.0)
- Transitioned to Meson build system
- Dropped CLI executables and external data payload
- Split into libopensps and libopensps-devel