module fsps_datalib
    !> @brief Access to data libraries in the standard format (doc/DATA_FORMAT.md).
    !>
    !> @details
    !> A library is `<root>/libraries/<kind>/<name>/manifest.json` plus one raw little-endian
    !> array per file. This module opens and validates a manifest, returns its metadata and axis
    !> values, and reads arrays (or contiguous blocks of them) into FP64 or integer buffers with
    !> stream I/O, checking each file's size against its declared dtype and shape.
    !>
    !> Shapes in a manifest are in C order (slowest first); the same bytes are a Fortran array
    !> with the dimensions reversed. The readers here take a flat buffer and an element offset,
    !> so callers pass their (contiguous) target arrays directly.
    !>
    !> Extra axes: an array may have leading axes beyond those its kind requires. The library
    !> name may carry selections, `name:axis=value[,axis=value]` (a numeric value or a label);
    !> `datalib_check_array` turns them into an element offset. An extra axis of length 1 needs
    !> no selection.
    !>
    !> Errors (missing files, malformed manifests, wrong sizes) stop the program with a message
    !> naming the file, as the legacy readers do.
    use fsps_precision, only: WP
    use fsps_json
    use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64, error_unit

    implicit none
    private

    public :: datalib_t
    public :: DATA_FORMAT_VERSION, DATA_ROOT_INDEX, DATA_ROOT_INDEX_OLD
    public :: DATA_FORMAT_AUTO, DATA_FORMAT_LEGACY, DATA_FORMAT_STANDARD
    public :: datalib_format_from_env, datalib_root_is_standard, datalib_available
    public :: datalib_open, datalib_base_name
    public :: datalib_has_meta, datalib_meta_real, datalib_meta_string
    public :: datalib_has_axis, datalib_axis_len, datalib_axis_values
    public :: datalib_has_array, datalib_array_len, datalib_check_array
    public :: datalib_read_real, datalib_read_int
    public :: datalib_meta_node

    integer, parameter :: DATA_FORMAT_VERSION = 1
    !> Root index and format identifier; the names before the libopensps rename ('fsps-data.json',
    !> 'fsps-data') are still read (transition alias, doc/DATA_FORMAT.md)
    character(len=*), parameter :: DATA_ROOT_INDEX = 'opensps-data.json'
    character(len=*), parameter :: DATA_ROOT_INDEX_OLD = 'fsps-data.json'
    character(len=*), parameter :: DATA_FORMAT_ID = 'opensps-data'
    character(len=*), parameter :: DATA_FORMAT_ID_OLD = 'fsps-data'
    character(len=*), parameter :: DATA_FORMAT_AUTO = 'auto'
    character(len=*), parameter :: DATA_FORMAT_LEGACY = 'legacy'
    character(len=*), parameter :: DATA_FORMAT_STANDARD = 'standard'
    integer, parameter :: MAX_SEL = 8

    type :: datalib_t
        character(len=:), allocatable :: dir     !< library directory (no trailing slash)
        character(len=:), allocatable :: kind
        character(len=:), allocatable :: name
        type(json_doc_t) :: doc
        integer :: metadata = 0, axes = 0, arrays = 0
        integer :: nsel = 0
        character(len=64) :: sel_axis(MAX_SEL) = ''
        character(len=64) :: sel_value(MAX_SEL) = ''
    end type datalib_t

contains

    ! ------------------------------------------------------------------------------------------
    ! Format selection
    ! ------------------------------------------------------------------------------------------

    !> The data format requested through FSPS_DATA_FORMAT: 'legacy', 'standard' or 'auto'
    !> (default: the standard format for every library a standard root provides).
    function datalib_format_from_env() result(mode)
        character(len=8) :: mode
        character(len=32) :: buf
        integer :: stat, n

        mode = DATA_FORMAT_AUTO
        call get_environment_variable('FSPS_DATA_FORMAT', value=buf, length=n, status=stat)
        if (stat /= 0 .or. n == 0) return
        select case (to_lower_local(trim(adjustl(buf))))
        case ('legacy')
            mode = DATA_FORMAT_LEGACY
        case ('standard', 'std')
            mode = DATA_FORMAT_STANDARD
        case ('auto', '')
            mode = DATA_FORMAT_AUTO
        case default
            write(error_unit, '(3A)') '[FSPS_DATA] Error: FSPS_DATA_FORMAT must be legacy, standard or auto, not "', &
                trim(buf), '"'
            error stop 1
        end select
    end function datalib_format_from_env

    !> True if `root` holds a standard-format root index (opensps-data.json, or the older
    !> fsps-data.json).
    logical function datalib_root_is_standard(root)
        character(len=*), intent(in) :: root
        inquire(file=trim(root) // '/' // DATA_ROOT_INDEX, exist=datalib_root_is_standard)
        if (.not. datalib_root_is_standard) &
            inquire(file=trim(root) // '/' // DATA_ROOT_INDEX_OLD, exist=datalib_root_is_standard)
    end function datalib_root_is_standard

    !> Whether library (kind, name) is read in the standard format under the given mode:
    !> 'legacy' never, 'standard' always (a missing library is then an error at open), 'auto'
    !> when its manifest exists.
    logical function datalib_available(root, kind, name, mode)
        character(len=*), intent(in) :: root, kind, name, mode
        select case (trim(mode))
        case (DATA_FORMAT_LEGACY)
            datalib_available = .false.
        case (DATA_FORMAT_STANDARD)
            datalib_available = .true.
        case default
            inquire(file=library_dir(root, kind, datalib_base_name(name)) // '/manifest.json', &
                    exist=datalib_available)
        end select
    end function datalib_available

    !> The library name without its axis selections (`c3k:afe=0.4` -> `c3k`).
    function datalib_base_name(spec) result(name)
        character(len=*), intent(in) :: spec
        character(len=:), allocatable :: name
        integer :: k
        k = index(spec, ':')
        if (k > 0) then
            name = trim(spec(1:k - 1))
        else
            name = trim(spec)
        end if
    end function datalib_base_name

    function library_dir(root, kind, name) result(dir)
        character(len=*), intent(in) :: root, kind, name
        character(len=:), allocatable :: dir
        dir = trim(root) // '/libraries/' // trim(kind) // '/' // trim(name)
    end function library_dir

    ! ------------------------------------------------------------------------------------------
    ! Opening a library
    ! ------------------------------------------------------------------------------------------

    !> Opens and validates `<root>/libraries/<kind>/<name>/manifest.json`. `spec` is the library
    !> name, optionally followed by axis selections (`name:axis=value,axis=value`).
    subroutine datalib_open(lib, root, kind, spec)
        type(datalib_t), intent(out) :: lib
        character(len=*), intent(in) :: root, kind, spec

        logical :: ok
        character(len=512) :: msg
        integer :: node, k, kc, ke
        character(len=:), allocatable :: sels, item, fmt

        lib%kind = trim(kind)
        lib%name = datalib_base_name(spec)
        lib%dir = library_dir(root, kind, lib%name)

        call json_parse_file(lib%doc, lib%dir // '/manifest.json', ok, msg)
        if (.not. ok) then
            write(error_unit, '(A)') '[FSPS_DATA] Error: cannot read the manifest of library ' // &
                trim(kind) // '/' // lib%name // ': ' // trim(msg)
            if (.not. datalib_root_is_standard(root)) write(error_unit, '(A)') &
                '           (' // trim(root) // ' has no ' // DATA_ROOT_INDEX // ': not a standard-format data root)'
            error stop 1
        end if

        fmt = json_string(lib%doc, json_child(lib%doc, 1, 'format'))
        if (fmt /= DATA_FORMAT_ID .and. fmt /= DATA_FORMAT_ID_OLD) &
            call lib_error(lib, 'manifest.json is not an opensps-data manifest (field "format")')
        node = json_child(lib%doc, 1, 'format_version')
        if (json_int(lib%doc, node) /= DATA_FORMAT_VERSION) &
            call lib_error(lib, 'unsupported format_version (this code reads version 1)')
        if (json_string(lib%doc, json_child(lib%doc, 1, 'kind')) /= lib%kind) &
            call lib_error(lib, 'manifest kind is "' // json_string(lib%doc, json_child(lib%doc, 1, 'kind')) // &
                                '", expected "' // lib%kind // '"')
        if (json_string(lib%doc, json_child(lib%doc, 1, 'name')) /= lib%name) &
            call lib_error(lib, 'manifest name differs from the directory name')

        lib%metadata = json_child(lib%doc, 1, 'metadata')
        lib%axes = json_child(lib%doc, 1, 'axes')
        lib%arrays = json_child(lib%doc, 1, 'arrays')
        if (json_kind(lib%doc, lib%axes) /= JSON_K_OBJECT) call lib_error(lib, 'missing "axes" object')
        if (json_kind(lib%doc, lib%arrays) /= JSON_K_OBJECT) call lib_error(lib, 'missing "arrays" object')

        ! Axis selections after ':'
        k = index(spec, ':')
        if (k > 0) then
            sels = trim(spec(k + 1:)) // ','
            do while (len(sels) > 0)
                kc = index(sels, ',')
                item = trim(adjustl(sels(1:kc - 1)))
                sels = sels(kc + 1:)
                if (len(item) == 0) cycle
                ke = index(item, '=')
                if (ke <= 1) call lib_error(lib, 'bad axis selection "' // item // '" (expected axis=value)')
                if (lib%nsel == MAX_SEL) call lib_error(lib, 'too many axis selections')
                lib%nsel = lib%nsel + 1
                lib%sel_axis(lib%nsel) = trim(item(1:ke - 1))
                lib%sel_value(lib%nsel) = trim(item(ke + 1:))
                if (.not. datalib_has_axis(lib, trim(lib%sel_axis(lib%nsel)))) &
                    call lib_error(lib, 'selection on unknown axis "' // trim(lib%sel_axis(lib%nsel)) // '"')
            end do
        end if
    end subroutine datalib_open

    subroutine lib_error(lib, msg)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: msg
        write(error_unit, '(A)') '[FSPS_DATA] Error in library ' // lib%kind // '/' // lib%name // &
            ' (' // lib%dir // '): ' // msg
        error stop 1
    end subroutine lib_error

    ! ------------------------------------------------------------------------------------------
    ! Metadata and axes
    ! ------------------------------------------------------------------------------------------

    logical function datalib_has_meta(lib, key)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: key
        datalib_has_meta = (json_child(lib%doc, lib%metadata, key) /= 0)
    end function datalib_has_meta

    !> The JSON node of metadata entry `key` (0 if absent), for structured metadata.
    integer function datalib_meta_node(lib, key)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: key
        datalib_meta_node = json_child(lib%doc, lib%metadata, key)
    end function datalib_meta_node

    real(WP) function datalib_meta_real(lib, key)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: key
        logical :: ok
        datalib_meta_real = json_real(lib%doc, json_child(lib%doc, lib%metadata, key), ok)
        if (.not. ok) call lib_error(lib, 'metadata "' // key // '" must be a number')
    end function datalib_meta_real

    function datalib_meta_string(lib, key) result(s)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: key
        character(len=:), allocatable :: s
        integer :: node
        node = json_child(lib%doc, lib%metadata, key)
        if (json_kind(lib%doc, node) /= JSON_K_STRING) call lib_error(lib, 'metadata "' // key // '" must be a string')
        s = json_string(lib%doc, node)
    end function datalib_meta_string

    logical function datalib_has_axis(lib, axis)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: axis
        datalib_has_axis = (json_child(lib%doc, lib%axes, axis) /= 0)
    end function datalib_has_axis

    integer function axis_node(lib, axis)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: axis
        axis_node = json_child(lib%doc, lib%axes, axis)
        if (axis_node == 0) call lib_error(lib, 'no axis "' // axis // '"')
    end function axis_node

    !> Number of coordinates of an axis.
    integer function datalib_axis_len(lib, axis)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: axis
        integer :: node
        integer(int64), allocatable :: dims(:)
        node = axis_node(lib, axis)
        if (json_child(lib%doc, node, 'labels') /= 0) then
            datalib_axis_len = json_len(lib%doc, json_child(lib%doc, node, 'labels'))
        else if (json_child(lib%doc, node, 'values') /= 0) then
            datalib_axis_len = json_len(lib%doc, json_child(lib%doc, node, 'values'))
        else
            call descriptor_dims(lib, node, 'axis ' // axis, dims)
            if (size(dims) /= 1) call lib_error(lib, 'axis "' // axis // '" must be one-dimensional')
            datalib_axis_len = int(dims(1))
        end if
    end function datalib_axis_len

    !> Coordinates of a numeric axis (inline `values` or a file).
    subroutine datalib_axis_values(lib, axis, vals)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: axis
        real(WP), allocatable, intent(out) :: vals(:)
        integer :: node, n
        logical :: ok

        node = axis_node(lib, axis)
        if (json_child(lib%doc, node, 'values') /= 0) then
            call json_real_array(lib%doc, json_child(lib%doc, node, 'values'), vals, ok)
            if (.not. ok) call lib_error(lib, 'axis "' // axis // '": values must be numbers')
        else if (json_child(lib%doc, node, 'file') /= 0) then
            n = datalib_axis_len(lib, axis)
            allocate(vals(n))
            call read_descriptor_real(lib, node, 'axis ' // axis, vals, int(n, int64), 0_int64)
        else
            call lib_error(lib, 'axis "' // axis // '" has no numeric coordinates')
        end if
    end subroutine datalib_axis_values

    ! ------------------------------------------------------------------------------------------
    ! Arrays
    ! ------------------------------------------------------------------------------------------

    logical function datalib_has_array(lib, name)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name
        datalib_has_array = (json_child(lib%doc, lib%arrays, name) /= 0)
    end function datalib_has_array

    integer function array_node(lib, name)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name
        array_node = json_child(lib%doc, lib%arrays, name)
        if (array_node == 0) call lib_error(lib, 'no array "' // name // '"')
    end function array_node

    !> Total number of elements of an array.
    integer(int64) function datalib_array_len(lib, name)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name
        integer(int64), allocatable :: dims(:)
        call descriptor_dims(lib, array_node(lib, name), 'array ' // name, dims)
        datalib_array_len = product(dims)
    end function datalib_array_len

    !> Checks that array `name` ends with the axes `axes` (in C order) of lengths `dims`, with
    !> any extra leading axes resolved by the selections, and returns the element offset of the
    !> selected block of product(dims) elements.
    subroutine datalib_check_array(lib, name, axes, dims, offset)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: axes(:)
        integer, intent(in) :: dims(:)
        integer(int64), intent(out) :: offset

        integer :: node, anode, nax, nreq, nextra, i, j, idx
        integer(int64), allocatable :: shape_(:)
        character(len=:), allocatable :: axname
        integer(int64) :: stride

        node = array_node(lib, name)
        call descriptor_dims(lib, node, 'array ' // name, shape_)
        anode = json_child(lib%doc, node, 'axes')
        nax = size(shape_)
        nreq = size(axes)
        if (json_kind(lib%doc, anode) /= JSON_K_ARRAY .or. json_len(lib%doc, anode) /= nax) &
            call lib_error(lib, 'array "' // name // '": "axes" must name each of its dimensions')
        if (nax < nreq) call lib_error(lib, 'array "' // name // '" has fewer dimensions than its kind requires')
        nextra = nax - nreq

        do i = 1, nreq
            axname = json_string(lib%doc, json_item(lib%doc, anode, nextra + i))
            if (axname /= trim(axes(i))) call lib_error(lib, 'array "' // name // '": dimension ' // &
                itoa(nextra + i) // ' is "' // axname // '", expected "' // trim(axes(i)) // '"')
            if (shape_(nextra + i) /= dims(i)) call lib_error(lib, 'array "' // name // '": axis "' // axname // &
                '" has ' // itoa(int(shape_(nextra + i))) // ' entries, expected ' // itoa(dims(i)))
            if (datalib_axis_len(lib, axname) /= shape_(nextra + i)) call lib_error(lib, 'array "' // name // &
                '": its dimension differs from the length of axis "' // axname // '"')
        end do

        ! Offset of the selected block: C order, the last extra axis varies fastest
        offset = 0
        stride = product(int(dims, int64))
        do i = nextra, 1, -1
            axname = json_string(lib%doc, json_item(lib%doc, anode, i))
            idx = 1
            if (shape_(i) > 1) then
                idx = 0
                do j = 1, lib%nsel
                    if (trim(lib%sel_axis(j)) == axname) idx = select_index(lib, axname, trim(lib%sel_value(j)))
                end do
                if (idx == 0) idx = default_index(lib, axname)
                if (idx == 0) call lib_error(lib, 'array "' // name // '" has an extra axis "' // axname // &
                    '" of length ' // itoa(int(shape_(i))) // ': select one value with "' // lib%name // ':' // &
                    axname // '=<value>"')
            end if
            offset = offset + int(idx - 1, int64) * stride
            stride = stride * shape_(i)
        end do
    end subroutine datalib_check_array

    !> Index of the axis's "default" coordinate (a number or a label), 0 if it has none.
    integer function default_index(lib, axis)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: axis
        integer :: node
        character(len=40) :: buf
        default_index = 0
        node = json_child(lib%doc, axis_node(lib, axis), 'default')
        select case (json_kind(lib%doc, node))
        case (JSON_K_NUMBER)
            write(buf, '(ES24.16)') json_real(lib%doc, node)
            default_index = select_index(lib, axis, trim(adjustl(buf)))
        case (JSON_K_STRING)
            default_index = select_index(lib, axis, json_string(lib%doc, node))
        end select
    end function default_index

    !> Index of `value` (a number or a label) on axis `axis`.
    integer function select_index(lib, axis, value)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: axis, value
        integer :: node, lab, i, ios
        real(WP), allocatable :: vals(:)
        real(WP) :: v

        select_index = 0
        node = axis_node(lib, axis)
        lab = json_child(lib%doc, node, 'labels')
        if (lab /= 0) then
            do i = 1, json_len(lib%doc, lab)
                if (json_string(lib%doc, json_item(lib%doc, lab, i)) == value) select_index = i
            end do
        else
            read(value, *, iostat=ios) v
            if (ios /= 0) call lib_error(lib, 'selection "' // axis // '=' // value // '" is not a number')
            call datalib_axis_values(lib, axis, vals)
            do i = 1, size(vals)
                if (abs(vals(i) - v) <= 1.0e-9_wp * max(1.0_wp, abs(v))) select_index = i
            end do
        end if
        if (select_index == 0) call lib_error(lib, 'axis "' // axis // '" has no value ' // value)
    end function select_index

    !> Reads n elements of array `name`, starting `offset` elements into the file, as FP64.
    !> FP32 data are promoted exactly.
    subroutine datalib_read_real(lib, name, n, out, offset)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name
        integer(int64), intent(in) :: n
        real(WP), intent(out) :: out(n)
        integer(int64), intent(in), optional :: offset
        integer(int64) :: off
        off = 0
        if (present(offset)) off = offset
        call read_descriptor_real(lib, array_node(lib, name), 'array ' // name, out, n, off)
    end subroutine datalib_read_real

    !> Reads n elements of an integer array.
    subroutine datalib_read_int(lib, name, n, out, offset)
        type(datalib_t), intent(in) :: lib
        character(len=*), intent(in) :: name
        integer(int64), intent(in) :: n
        integer, intent(out) :: out(n)
        integer(int64), intent(in), optional :: offset

        integer :: node, u
        integer(int64) :: off
        character(len=:), allocatable :: dtype, path
        integer(int32), allocatable :: b4(:)
        integer(int64), allocatable :: b8(:)

        off = 0
        if (present(offset)) off = offset
        node = array_node(lib, name)
        call open_descriptor(lib, node, 'array ' // name, n, off, dtype, path, u)
        select case (dtype)
        case ('<i4')
            allocate(b4(n))
            call read_block_i4(u, off, b4, path)
            out = b4
        case ('<i8')
            allocate(b8(n))
            call read_block_i8(u, off, b8, path)
            if (any(abs(b8) > huge(1_int32))) call lib_error(lib, 'array "' // name // '": value out of range')
            out = int(b8)
        case default
            call lib_error(lib, 'array "' // name // '" must be an integer array (<i4 or <i8)')
        end select
        close(u)
    end subroutine datalib_read_int

    ! ------------------------------------------------------------------------------------------
    ! Internals: descriptors (file, dtype, shape) and raw reads
    ! ------------------------------------------------------------------------------------------

    subroutine descriptor_dims(lib, node, what, dims)
        type(datalib_t), intent(in) :: lib
        integer, intent(in) :: node
        character(len=*), intent(in) :: what
        integer(int64), allocatable, intent(out) :: dims(:)
        logical :: ok
        call json_int_array(lib%doc, json_child(lib%doc, node, 'shape'), dims, ok)
        if (.not. ok .or. size(dims) == 0) call lib_error(lib, what // ': "shape" must be a list of integers')
        if (any(dims < 0)) call lib_error(lib, what // ': negative dimension')
    end subroutine descriptor_dims

    integer function itemsize(dtype)
        character(len=*), intent(in) :: dtype
        select case (dtype)
        case ('<f4', '<i4')
            itemsize = 4
        case ('<f8', '<i8')
            itemsize = 8
        case default
            itemsize = 0
        end select
    end function itemsize

    !> Opens the file of a descriptor for stream reading after checking its size and the range.
    subroutine open_descriptor(lib, node, what, n, offset, dtype, path, u)
        type(datalib_t), intent(in) :: lib
        integer, intent(in) :: node
        character(len=*), intent(in) :: what
        integer(int64), intent(in) :: n, offset
        character(len=:), allocatable, intent(out) :: dtype, path
        integer, intent(out) :: u

        integer(int64), allocatable :: dims(:)
        integer(int64) :: total, fsize
        integer :: ios
        logical :: exists

        dtype = json_string(lib%doc, json_child(lib%doc, node, 'dtype'))
        if (itemsize(dtype) == 0) call lib_error(lib, what // ': unsupported dtype "' // dtype // &
                                                 '" (use <f4, <f8, <i4 or <i8)')
        if (json_kind(lib%doc, json_child(lib%doc, node, 'file')) /= JSON_K_STRING) &
            call lib_error(lib, what // ': missing "file"')
        path = lib%dir // '/' // json_string(lib%doc, json_child(lib%doc, node, 'file'))
        call descriptor_dims(lib, node, what, dims)
        total = product(dims)
        if (offset < 0 .or. offset + n > total) call lib_error(lib, what // ': read beyond the array')

        inquire(file=path, exist=exists, size=fsize)
        if (.not. exists) call lib_error(lib, what // ': file ' // path // ' is missing')
        if (fsize /= total * itemsize(dtype)) call lib_error(lib, what // ': file ' // path // ' has ' // &
            itoa64(fsize) // ' bytes, expected ' // itoa64(total * itemsize(dtype)) // ' (truncated download?)')
        open(newunit=u, file=path, status='old', access='stream', form='unformatted', action='read', iostat=ios)
        if (ios /= 0) call lib_error(lib, what // ': cannot open ' // path)
    end subroutine open_descriptor

    subroutine read_descriptor_real(lib, node, what, out, n, offset)
        type(datalib_t), intent(in) :: lib
        integer, intent(in) :: node
        character(len=*), intent(in) :: what
        integer(int64), intent(in) :: n, offset
        real(WP), intent(out) :: out(n)

        integer :: u
        character(len=:), allocatable :: dtype, path
        real(real32), allocatable :: b4(:)
        real(real64), allocatable :: b8(:)

        call open_descriptor(lib, node, what, n, offset, dtype, path, u)
        select case (dtype)
        case ('<f4')
            allocate(b4(n))
            call read_block_r4(u, offset, b4, path)
            out = real(b4, WP)
        case ('<f8')
            allocate(b8(n))
            call read_block_r8(u, offset, b8, path)
            out = real(b8, WP)
        case default
            call lib_error(lib, what // ' must be a floating-point array (<f4 or <f8)')
        end select
        close(u)
    end subroutine read_descriptor_real

    logical function host_is_big_endian()
        host_is_big_endian = (transfer(1_int32, 1_int8) /= 1_int8)
    end function host_is_big_endian

    subroutine read_block_r4(u, offset, buf, path)
        integer, intent(in) :: u
        integer(int64), intent(in) :: offset
        real(real32), intent(out) :: buf(:)
        character(len=*), intent(in) :: path
        integer :: ios
        integer(int32), allocatable :: w(:)
        read(u, pos=offset * 4 + 1, iostat=ios) buf
        call check_read(ios, path)
        if (host_is_big_endian()) then
            w = transfer(buf, w)
            call swap4(w)
            buf = transfer(w, buf)
        end if
    end subroutine read_block_r4

    subroutine read_block_r8(u, offset, buf, path)
        integer, intent(in) :: u
        integer(int64), intent(in) :: offset
        real(real64), intent(out) :: buf(:)
        character(len=*), intent(in) :: path
        integer :: ios
        integer(int64), allocatable :: w(:)
        read(u, pos=offset * 8 + 1, iostat=ios) buf
        call check_read(ios, path)
        if (host_is_big_endian()) then
            w = transfer(buf, w)
            call swap8(w)
            buf = transfer(w, buf)
        end if
    end subroutine read_block_r8

    subroutine read_block_i4(u, offset, buf, path)
        integer, intent(in) :: u
        integer(int64), intent(in) :: offset
        integer(int32), intent(out) :: buf(:)
        character(len=*), intent(in) :: path
        integer :: ios
        read(u, pos=offset * 4 + 1, iostat=ios) buf
        call check_read(ios, path)
        if (host_is_big_endian()) call swap4(buf)
    end subroutine read_block_i4

    subroutine read_block_i8(u, offset, buf, path)
        integer, intent(in) :: u
        integer(int64), intent(in) :: offset
        integer(int64), intent(out) :: buf(:)
        character(len=*), intent(in) :: path
        integer :: ios
        read(u, pos=offset * 8 + 1, iostat=ios) buf
        call check_read(ios, path)
        if (host_is_big_endian()) call swap8(buf)
    end subroutine read_block_i8

    subroutine check_read(ios, path)
        integer, intent(in) :: ios
        character(len=*), intent(in) :: path
        if (ios /= 0) then
            write(error_unit, '(A)') '[FSPS_DATA] Error reading ' // trim(path)
            error stop 1
        end if
    end subroutine check_read

    elemental subroutine swap4(w)
        integer(int32), intent(inout) :: w
        integer(int32) :: r
        integer :: k
        r = 0
        do k = 0, 3
            call mvbits(w, 8 * k, 8, r, 8 * (3 - k))
        end do
        w = r
    end subroutine swap4

    elemental subroutine swap8(w)
        integer(int64), intent(inout) :: w
        integer(int64) :: r
        integer :: k
        r = 0
        do k = 0, 7
            call mvbits(w, 8 * k, 8, r, 8 * (7 - k))
        end do
        w = r
    end subroutine swap8

    function itoa(i) result(s)
        integer, intent(in) :: i
        character(len=:), allocatable :: s
        character(len=24) :: b
        write(b, '(I0)') i
        s = trim(b)
    end function itoa

    function itoa64(i) result(s)
        integer(int64), intent(in) :: i
        character(len=:), allocatable :: s
        character(len=24) :: b
        write(b, '(I0)') i
        s = trim(b)
    end function itoa64

    pure function to_lower_local(s) result(r)
        character(len=*), intent(in) :: s
        character(len=len(s)) :: r
        integer :: i, c
        r = s
        do i = 1, len(s)
            c = iachar(s(i:i))
            if (c >= 65 .and. c <= 90) r(i:i) = achar(c + 32)
        end do
    end function to_lower_local

end module fsps_datalib
