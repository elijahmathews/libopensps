module fsps_json
    !> @brief A small, complete JSON (RFC 8259) reader for the data-library manifests.
    !>
    !> @details
    !> Parses a document into a flat node table: every value is a node with its kind, its
    !> children (for arrays and objects, in document order), its key (for object members) and its
    !> scalar value. Strings are decoded (all escapes, including \uXXXX to UTF-8) into one string
    !> pool. Numbers are converted with a list-directed read, which rounds correctly, so a number
    !> written by Python's `json` (shortest round-trip form) reads back to the same FP64 value.
    !>
    !> The reader is used once per library at setup; it is not thread-safe on a shared document
    !> but holds no module state, so separate documents may be parsed concurrently.
    use fsps_precision, only: WP
    use, intrinsic :: iso_fortran_env, only: int64, iostat_end

    implicit none
    private

    public :: json_doc_t
    public :: json_parse_file, json_parse_string
    public :: json_child, json_item, json_len, json_kind, json_key
    public :: json_string, json_real, json_int, json_real_array, json_int_array, json_is_integer
    public :: JSON_K_NULL, JSON_K_FALSE, JSON_K_TRUE, JSON_K_NUMBER, JSON_K_STRING, JSON_K_ARRAY, JSON_K_OBJECT

    integer, parameter :: JSON_K_NULL = 0, JSON_K_FALSE = 1, JSON_K_TRUE = 2, JSON_K_NUMBER = 3, &
                          JSON_K_STRING = 4, JSON_K_ARRAY = 5, JSON_K_OBJECT = 6
    integer, parameter :: MAX_DEPTH = 64
    !> Backslash as achar(92): some compilers (nvfortran) treat it as an escape in literals
    character(len=1), parameter :: BSLASH = achar(92)

    type :: json_node_t
        integer :: kind = JSON_K_NULL
        integer :: first = 0          !< first child (arrays, objects)
        integer :: last = 0           !< last child
        integer :: next = 0           !< next sibling
        integer :: nchild = 0
        integer :: key_lo = 1, key_len = 0   !< key in the string pool (object members)
        integer :: str_lo = 1, str_len = 0   !< value in the string pool (strings)
        logical :: is_int = .false.          !< number written without fraction or exponent
        integer(int64) :: ival = 0
        real(WP) :: rval = 0.0_wp
    end type json_node_t

    !> A parsed document. Node 1 is the root value.
    type :: json_doc_t
        type(json_node_t), allocatable :: nodes(:)
        integer :: n = 0
        character(len=:), allocatable :: pool
        integer :: pool_len = 0
    end type json_doc_t

    !> Parser state (one per parse)
    type :: parser_t
        character(len=:), allocatable :: text
        integer :: pos = 1
        integer :: depth = 0
        logical :: failed = .false.
        character(len=256) :: msg = ''
    end type parser_t

contains

    ! ------------------------------------------------------------------------------------------
    ! Entry points
    ! ------------------------------------------------------------------------------------------

    !> Parses the JSON file `path`; on failure `ok` is false and `errmsg` says where and why.
    subroutine json_parse_file(doc, path, ok, errmsg)
        type(json_doc_t), intent(out) :: doc
        character(len=*), intent(in) :: path
        logical, intent(out) :: ok
        character(len=*), intent(out) :: errmsg

        integer :: u, ios, nbytes
        character(len=:), allocatable :: text

        ok = .false.
        errmsg = ''
        open(newunit=u, file=trim(path), status='old', access='stream', form='unformatted', &
             action='read', iostat=ios)
        if (ios /= 0) then
            errmsg = 'cannot open ' // trim(path)
            return
        end if
        inquire(unit=u, size=nbytes)
        if (nbytes < 0) then
            close(u)
            errmsg = 'cannot determine the size of ' // trim(path)
            return
        end if
        allocate(character(len=nbytes) :: text)
        if (nbytes > 0) read(u, iostat=ios) text
        close(u)
        if (ios /= 0) then
            errmsg = 'cannot read ' // trim(path)
            return
        end if
        call json_parse_string(doc, text, ok, errmsg)
        if (.not. ok) errmsg = trim(path) // ': ' // trim(errmsg)
    end subroutine json_parse_file

    !> Parses a JSON document held in a string.
    subroutine json_parse_string(doc, text, ok, errmsg)
        type(json_doc_t), intent(out) :: doc
        character(len=*), intent(in) :: text
        logical, intent(out) :: ok
        character(len=*), intent(out) :: errmsg

        type(parser_t) :: p
        integer :: root

        allocate(doc%nodes(64))
        doc%n = 0
        allocate(character(len=max(256, len(text) / 4)) :: doc%pool)
        doc%pool_len = 0

        p%text = text
        p%pos = 1
        call skip_ws(p)
        ! A UTF-8 byte-order mark is tolerated
        if (len(p%text) >= 3) then
            if (p%text(1:3) == achar(239) // achar(187) // achar(191)) p%pos = 4
        end if
        call parse_value(p, doc, root)
        if (.not. p%failed) then
            call skip_ws(p)
            if (p%pos <= len(p%text)) call fail(p, 'unexpected text after the document')
        end if
        ok = .not. p%failed
        errmsg = ''
        if (.not. ok) then
            write(errmsg, '(A, " (at byte ", I0, ")")') trim(p%msg), p%pos
        end if
    end subroutine json_parse_string

    ! ------------------------------------------------------------------------------------------
    ! Queries
    ! ------------------------------------------------------------------------------------------

    integer function json_kind(doc, node)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        json_kind = -1
        if (node >= 1 .and. node <= doc%n) json_kind = doc%nodes(node)%kind
    end function json_kind

    !> Number of elements of an array or members of an object (0 for other values).
    integer function json_len(doc, node)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        json_len = 0
        if (node >= 1 .and. node <= doc%n) json_len = doc%nodes(node)%nchild
    end function json_len

    !> The i-th (1-based) element of an array or member of an object, 0 if absent.
    integer function json_item(doc, node, i)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node, i
        integer :: c, k
        json_item = 0
        if (node < 1 .or. node > doc%n) return
        if (i < 1 .or. i > doc%nodes(node)%nchild) return
        c = doc%nodes(node)%first
        do k = 2, i
            c = doc%nodes(c)%next
        end do
        json_item = c
    end function json_item

    !> The member `key` of an object, 0 if absent or if `node` is not an object.
    integer function json_child(doc, node, key)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        character(len=*), intent(in) :: key
        integer :: c
        json_child = 0
        if (node < 1 .or. node > doc%n) return
        if (doc%nodes(node)%kind /= JSON_K_OBJECT) return
        c = doc%nodes(node)%first
        do while (c /= 0)
            if (doc%nodes(c)%key_len == len(key)) then
                if (doc%pool(doc%nodes(c)%key_lo:doc%nodes(c)%key_lo + len(key) - 1) == key) then
                    json_child = c
                    return
                end if
            end if
            c = doc%nodes(c)%next
        end do
    end function json_child

    !> The key of an object member.
    function json_key(doc, node) result(s)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        character(len=:), allocatable :: s
        s = ''
        if (node < 1 .or. node > doc%n) return
        s = doc%pool(doc%nodes(node)%key_lo:doc%nodes(node)%key_lo + doc%nodes(node)%key_len - 1)
    end function json_key

    !> The value of a string node ('' for other kinds).
    function json_string(doc, node) result(s)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        character(len=:), allocatable :: s
        s = ''
        if (json_kind(doc, node) /= JSON_K_STRING) return
        s = doc%pool(doc%nodes(node)%str_lo:doc%nodes(node)%str_lo + doc%nodes(node)%str_len - 1)
    end function json_string

    logical function json_is_integer(doc, node)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        json_is_integer = .false.
        if (json_kind(doc, node) == JSON_K_NUMBER) json_is_integer = doc%nodes(node)%is_int
    end function json_is_integer

    !> The value of a number node; `ok` is false for other kinds.
    real(WP) function json_real(doc, node, ok)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        logical, intent(out), optional :: ok
        json_real = 0.0_wp
        if (present(ok)) ok = (json_kind(doc, node) == JSON_K_NUMBER)
        if (json_kind(doc, node) == JSON_K_NUMBER) json_real = doc%nodes(node)%rval
    end function json_real

    !> The value of an integer number node; `ok` is false otherwise.
    integer(int64) function json_int(doc, node, ok)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        logical, intent(out), optional :: ok
        json_int = 0
        if (present(ok)) ok = json_is_integer(doc, node)
        if (json_is_integer(doc, node)) json_int = doc%nodes(node)%ival
    end function json_int

    !> An array of numbers as FP64; `ok` is false if `node` is not such an array.
    subroutine json_real_array(doc, node, vals, ok)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        real(WP), allocatable, intent(out) :: vals(:)
        logical, intent(out) :: ok
        integer :: c, i
        ok = .false.
        if (json_kind(doc, node) /= JSON_K_ARRAY) return
        allocate(vals(doc%nodes(node)%nchild))
        c = doc%nodes(node)%first
        do i = 1, size(vals)
            if (doc%nodes(c)%kind /= JSON_K_NUMBER) return
            vals(i) = doc%nodes(c)%rval
            c = doc%nodes(c)%next
        end do
        ok = .true.
    end subroutine json_real_array

    !> An array of integers; `ok` is false if `node` is not such an array.
    subroutine json_int_array(doc, node, vals, ok)
        type(json_doc_t), intent(in) :: doc
        integer, intent(in) :: node
        integer(int64), allocatable, intent(out) :: vals(:)
        logical, intent(out) :: ok
        integer :: c, i
        ok = .false.
        if (json_kind(doc, node) /= JSON_K_ARRAY) return
        allocate(vals(doc%nodes(node)%nchild))
        c = doc%nodes(node)%first
        do i = 1, size(vals)
            if (.not. json_is_integer(doc, c)) return
            vals(i) = doc%nodes(c)%ival
            c = doc%nodes(c)%next
        end do
        ok = .true.
    end subroutine json_int_array

    ! ------------------------------------------------------------------------------------------
    ! Parser
    ! ------------------------------------------------------------------------------------------

    subroutine fail(p, msg)
        type(parser_t), intent(inout) :: p
        character(len=*), intent(in) :: msg
        if (.not. p%failed) p%msg = msg
        p%failed = .true.
    end subroutine fail

    subroutine skip_ws(p)
        type(parser_t), intent(inout) :: p
        do while (p%pos <= len(p%text))
            select case (iachar(p%text(p%pos:p%pos)))
            case (32, 9, 10, 13)
                p%pos = p%pos + 1
            case default
                return
            end select
        end do
    end subroutine skip_ws

    integer function new_node(doc, kind)
        type(json_doc_t), intent(inout) :: doc
        integer, intent(in) :: kind
        type(json_node_t), allocatable :: tmp(:)
        if (doc%n == size(doc%nodes)) then
            allocate(tmp(2 * size(doc%nodes)))
            tmp(1:doc%n) = doc%nodes(1:doc%n)
            call move_alloc(tmp, doc%nodes)
        end if
        doc%n = doc%n + 1
        doc%nodes(doc%n) = json_node_t()
        doc%nodes(doc%n)%kind = kind
        new_node = doc%n
    end function new_node

    subroutine add_child(doc, parent, child)
        type(json_doc_t), intent(inout) :: doc
        integer, intent(in) :: parent, child
        if (doc%nodes(parent)%first == 0) then
            doc%nodes(parent)%first = child
        else
            doc%nodes(doc%nodes(parent)%last)%next = child
        end if
        doc%nodes(parent)%last = child
        doc%nodes(parent)%nchild = doc%nodes(parent)%nchild + 1
    end subroutine add_child

    subroutine pool_append(doc, s)
        type(json_doc_t), intent(inout) :: doc
        character(len=*), intent(in) :: s
        character(len=:), allocatable :: tmp
        if (doc%pool_len + len(s) > len(doc%pool)) then
            allocate(character(len=2 * (doc%pool_len + len(s))) :: tmp)
            tmp(1:doc%pool_len) = doc%pool(1:doc%pool_len)
            call move_alloc(tmp, doc%pool)
        end if
        doc%pool(doc%pool_len + 1:doc%pool_len + len(s)) = s
        doc%pool_len = doc%pool_len + len(s)
    end subroutine pool_append

    recursive subroutine parse_value(p, doc, node)
        type(parser_t), intent(inout) :: p
        type(json_doc_t), intent(inout) :: doc
        integer, intent(out) :: node
        character :: c
        integer :: str_lo, str_len

        node = 0
        call skip_ws(p)
        if (p%pos > len(p%text)) then
            call fail(p, 'unexpected end of document')
            return
        end if
        c = p%text(p%pos:p%pos)
        select case (c)
        case ('{')
            call parse_object(p, doc, node)
        case ('[')
            call parse_array(p, doc, node)
        case ('"')
            node = new_node(doc, JSON_K_STRING)
            str_lo = doc%pool_len + 1
            call parse_string(p, doc, str_len)
            doc%nodes(node)%str_lo = str_lo
            doc%nodes(node)%str_len = str_len
        case ('t')
            call parse_literal(p, 'true')
            node = new_node(doc, JSON_K_TRUE)
        case ('f')
            call parse_literal(p, 'false')
            node = new_node(doc, JSON_K_FALSE)
        case ('n')
            call parse_literal(p, 'null')
            node = new_node(doc, JSON_K_NULL)
        case ('-', '0':'9')
            node = new_node(doc, JSON_K_NUMBER)
            call parse_number(p, doc, node)
        case default
            call fail(p, 'unexpected character "' // c // '"')
        end select
    end subroutine parse_value

    subroutine parse_literal(p, word)
        type(parser_t), intent(inout) :: p
        character(len=*), intent(in) :: word
        if (p%pos + len(word) - 1 > len(p%text)) then
            call fail(p, 'invalid literal')
        else if (p%text(p%pos:p%pos + len(word) - 1) /= word) then
            call fail(p, 'invalid literal')
        else
            p%pos = p%pos + len(word)
        end if
    end subroutine parse_literal

    recursive subroutine parse_object(p, doc, node)
        type(parser_t), intent(inout) :: p
        type(json_doc_t), intent(inout) :: doc
        integer, intent(out) :: node
        integer :: child, key_lo, key_len

        node = new_node(doc, JSON_K_OBJECT)
        p%depth = p%depth + 1
        if (p%depth > MAX_DEPTH) then
            call fail(p, 'nesting too deep')
            return
        end if
        p%pos = p%pos + 1                        ! '{'
        call skip_ws(p)
        if (p%pos <= len(p%text)) then
            if (p%text(p%pos:p%pos) == '}') then
                p%pos = p%pos + 1
                p%depth = p%depth - 1
                return
            end if
        end if
        do
            call skip_ws(p)
            if (p%pos > len(p%text)) then
                call fail(p, 'unterminated object')
                return
            end if
            if (p%text(p%pos:p%pos) /= '"') then
                call fail(p, 'expected a member name')
                return
            end if
            key_lo = doc%pool_len + 1
            call parse_string(p, doc, key_len)
            if (p%failed) return
            call skip_ws(p)
            if (p%pos > len(p%text)) then
                call fail(p, 'unterminated object')
                return
            end if
            if (p%text(p%pos:p%pos) /= ':') then
                call fail(p, 'expected ":"')
                return
            end if
            p%pos = p%pos + 1
            call parse_value(p, doc, child)
            if (p%failed) return
            doc%nodes(child)%key_lo = key_lo
            doc%nodes(child)%key_len = key_len
            call add_child(doc, node, child)
            call skip_ws(p)
            if (p%pos > len(p%text)) then
                call fail(p, 'unterminated object')
                return
            end if
            select case (p%text(p%pos:p%pos))
            case (',')
                p%pos = p%pos + 1
            case ('}')
                p%pos = p%pos + 1
                exit
            case default
                call fail(p, 'expected "," or "}"')
                return
            end select
        end do
        p%depth = p%depth - 1
    end subroutine parse_object

    recursive subroutine parse_array(p, doc, node)
        type(parser_t), intent(inout) :: p
        type(json_doc_t), intent(inout) :: doc
        integer, intent(out) :: node
        integer :: child

        node = new_node(doc, JSON_K_ARRAY)
        p%depth = p%depth + 1
        if (p%depth > MAX_DEPTH) then
            call fail(p, 'nesting too deep')
            return
        end if
        p%pos = p%pos + 1                        ! '['
        call skip_ws(p)
        if (p%pos <= len(p%text)) then
            if (p%text(p%pos:p%pos) == ']') then
                p%pos = p%pos + 1
                p%depth = p%depth - 1
                return
            end if
        end if
        do
            call parse_value(p, doc, child)
            if (p%failed) return
            call add_child(doc, node, child)
            call skip_ws(p)
            if (p%pos > len(p%text)) then
                call fail(p, 'unterminated array')
                return
            end if
            select case (p%text(p%pos:p%pos))
            case (',')
                p%pos = p%pos + 1
            case (']')
                p%pos = p%pos + 1
                exit
            case default
                call fail(p, 'expected "," or "]"')
                return
            end select
        end do
        p%depth = p%depth - 1
    end subroutine parse_array

    !> Decodes a string starting at the opening quote into the pool; returns its decoded length.
    subroutine parse_string(p, doc, slen)
        type(parser_t), intent(inout) :: p
        type(json_doc_t), intent(inout) :: doc
        integer, intent(out) :: slen
        integer :: start, code, low, ios
        character :: c

        slen = 0
        start = doc%pool_len
        p%pos = p%pos + 1                        ! opening quote
        do
            if (p%pos > len(p%text)) then
                call fail(p, 'unterminated string')
                return
            end if
            c = p%text(p%pos:p%pos)
            if (c == '"') then
                p%pos = p%pos + 1
                exit
            else if (c == BSLASH) then
                if (p%pos + 1 > len(p%text)) then
                    call fail(p, 'unterminated string')
                    return
                end if
                select case (p%text(p%pos + 1:p%pos + 1))
                case ('"')
                    call pool_append(doc, '"')
                case (BSLASH)
                    call pool_append(doc, BSLASH)
                case ('/')
                    call pool_append(doc, '/')
                case ('b')
                    call pool_append(doc, achar(8))
                case ('f')
                    call pool_append(doc, achar(12))
                case ('n')
                    call pool_append(doc, achar(10))
                case ('r')
                    call pool_append(doc, achar(13))
                case ('t')
                    call pool_append(doc, achar(9))
                case ('u')
                    if (p%pos + 5 > len(p%text)) then
                        call fail(p, 'bad unicode escape')
                        return
                    end if
                    read(p%text(p%pos + 2:p%pos + 5), '(Z4)', iostat=ios) code
                    if (ios /= 0) then
                        call fail(p, 'bad unicode escape')
                        return
                    end if
                    p%pos = p%pos + 4
                    ! Surrogate pair
                    if (code >= 55296 .and. code <= 56319) then
                        if (p%pos + 7 > len(p%text)) then
                            call fail(p, 'bad surrogate pair')
                            return
                        end if
                        if (p%text(p%pos + 2:p%pos + 3) /= BSLASH // 'u') then
                            call fail(p, 'bad surrogate pair')
                            return
                        end if
                        read(p%text(p%pos + 4:p%pos + 7), '(Z4)', iostat=ios) low
                        if (ios /= 0 .or. low < 56320 .or. low > 57343) then
                            call fail(p, 'bad surrogate pair')
                            return
                        end if
                        code = 65536 + (code - 55296) * 1024 + (low - 56320)
                        p%pos = p%pos + 6
                    end if
                    call append_utf8(doc, code)
                case default
                    call fail(p, 'bad escape')
                    return
                end select
                p%pos = p%pos + 2
            else if (iachar(c) < 32) then
                call fail(p, 'control character in string')
                return
            else
                call pool_append(doc, c)
                p%pos = p%pos + 1
            end if
        end do
        slen = doc%pool_len - start
    end subroutine parse_string

    subroutine append_utf8(doc, code)
        type(json_doc_t), intent(inout) :: doc
        integer, intent(in) :: code
        if (code < 128) then
            call pool_append(doc, achar(code))
        else if (code < 2048) then
            call pool_append(doc, achar(192 + code / 64) // achar(128 + iand(code, 63)))
        else if (code < 65536) then
            call pool_append(doc, achar(224 + code / 4096) // achar(128 + iand(code / 64, 63)) // &
                                  achar(128 + iand(code, 63)))
        else
            call pool_append(doc, achar(240 + code / 262144) // achar(128 + iand(code / 4096, 63)) // &
                                  achar(128 + iand(code / 64, 63)) // achar(128 + iand(code, 63)))
        end if
    end subroutine append_utf8

    !> Number grammar: -? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?
    subroutine parse_number(p, doc, node)
        type(parser_t), intent(inout) :: p
        type(json_doc_t), intent(inout) :: doc
        integer, intent(in) :: node
        integer :: start, ios
        logical :: is_int

        start = p%pos
        is_int = .true.
        if (p%text(p%pos:p%pos) == '-') p%pos = p%pos + 1
        if (p%pos > len(p%text)) then
            call fail(p, 'bad number')
            return
        end if
        if (p%text(p%pos:p%pos) == '0') then
            p%pos = p%pos + 1
        else if (is_digit(p%text(p%pos:p%pos))) then
            do while (p%pos <= len(p%text))
                if (.not. is_digit(p%text(p%pos:p%pos))) exit
                p%pos = p%pos + 1
            end do
        else
            call fail(p, 'bad number')
            return
        end if
        if (p%pos <= len(p%text)) then
            if (p%text(p%pos:p%pos) == '.') then
                is_int = .false.
                p%pos = p%pos + 1
                if (.not. digits(p)) then
                    call fail(p, 'bad number: no digits after "."')
                    return
                end if
            end if
        end if
        if (p%pos <= len(p%text)) then
            if (p%text(p%pos:p%pos) == 'e' .or. p%text(p%pos:p%pos) == 'E') then
                is_int = .false.
                p%pos = p%pos + 1
                if (p%pos <= len(p%text)) then
                    if (p%text(p%pos:p%pos) == '+' .or. p%text(p%pos:p%pos) == '-') p%pos = p%pos + 1
                end if
                if (.not. digits(p)) then
                    call fail(p, 'bad number: no exponent digits')
                    return
                end if
            end if
        end if
        ! List-directed conversion rounds correctly (round to nearest)
        read(p%text(start:p%pos - 1), *, iostat=ios) doc%nodes(node)%rval
        if (ios /= 0) then
            call fail(p, 'number out of range: ' // p%text(start:p%pos - 1))
            return
        end if
        if (is_int .and. p%pos - start <= 18) then
            read(p%text(start:p%pos - 1), *, iostat=ios) doc%nodes(node)%ival
            doc%nodes(node)%is_int = (ios == 0)
        end if
    end subroutine parse_number

    logical function digits(p)
        type(parser_t), intent(inout) :: p
        integer :: start
        start = p%pos
        do while (p%pos <= len(p%text))
            if (.not. is_digit(p%text(p%pos:p%pos))) exit
            p%pos = p%pos + 1
        end do
        digits = (p%pos > start)
    end function digits

    pure logical function is_digit(c)
        character, intent(in) :: c
        is_digit = (c >= '0' .and. c <= '9')
    end function is_digit

end module fsps_json
