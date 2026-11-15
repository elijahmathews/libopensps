module opensps_sha256
    !> @brief
    !> SHA-256 (FIPS 180-4), self-contained, for content hashes of bandpass curves.
    !>
    !> @details
    !> Streaming interface: `sha256_init`, any number of `sha256_update_*`, `sha256_hex`. Numbers
    !> are fed as explicit little-endian bytes (built with shifts, so the digest does not depend on
    !> the byte order of the machine). 32-bit words are held in `int64` and masked, which keeps the
    !> arithmetic exact without unsigned integers. Host code only.
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    implicit none
    private

    public :: sha256_t, sha256_init, sha256_update_bytes, sha256_update_int64, sha256_update_real64, &
              sha256_update_string, sha256_hex

    type :: sha256_t
        integer(int64) :: h(8) = 0_int64
        integer :: buf(64) = 0          !< pending bytes (0..255)
        integer :: nbuf = 0
        integer(int64) :: nbytes = 0_int64
    end type sha256_t

    integer(int64), parameter :: MASK32 = 4294967295_int64

    integer(int64), parameter :: K(64) = [ &
        1116352408_int64, 1899447441_int64, 3049323471_int64, 3921009573_int64, &
        961987163_int64, 1508970993_int64, 2453635748_int64, 2870763221_int64, &
        3624381080_int64, 310598401_int64, 607225278_int64, 1426881987_int64, &
        1925078388_int64, 2162078206_int64, 2614888103_int64, 3248222580_int64, &
        3835390401_int64, 4022224774_int64, 264347078_int64, 604807628_int64, &
        770255983_int64, 1249150122_int64, 1555081692_int64, 1996064986_int64, &
        2554220882_int64, 2821834349_int64, 2952996808_int64, 3210313671_int64, &
        3336571891_int64, 3584528711_int64, 113926993_int64, 338241895_int64, &
        666307205_int64, 773529912_int64, 1294757372_int64, 1396182291_int64, &
        1695183700_int64, 1986661051_int64, 2177026350_int64, 2456956037_int64, &
        2730485921_int64, 2820302411_int64, 3259730800_int64, 3345764771_int64, &
        3516065817_int64, 3600352804_int64, 4094571909_int64, 275423344_int64, &
        430227734_int64, 506948616_int64, 659060556_int64, 883997877_int64, &
        958139571_int64, 1322822218_int64, 1537002063_int64, 1747873779_int64, &
        1955562222_int64, 2024104815_int64, 2227730452_int64, 2361852424_int64, &
        2428436474_int64, 2756734187_int64, 3204031479_int64, 3329325298_int64]

contains

    pure function rotr(x, n) result(r)
        integer(int64), intent(in) :: x
        integer, intent(in) :: n
        integer(int64) :: r
        r = iand(ior(ishft(x, -n), ishft(x, 32 - n)), MASK32)
    end function rotr

    subroutine sha256_init(s)
        type(sha256_t), intent(out) :: s
        s%h = [1779033703_int64, 3144134277_int64, 1013904242_int64, 2773480762_int64, &
               1359893119_int64, 2600822924_int64, 528734635_int64, 1541459225_int64]
        s%buf = 0
        s%nbuf = 0
        s%nbytes = 0_int64
    end subroutine sha256_init

    !> One 64-byte block of s%buf
    subroutine compress(s)
        type(sha256_t), intent(inout) :: s
        integer(int64) :: w(64), a, b, c, d, e, f, g, hh, t1, t2, s0, s1
        integer :: i

        do i = 1, 16
            w(i) = ior(ior(ishft(int(s%buf(4*i - 3), int64), 24), ishft(int(s%buf(4*i - 2), int64), 16)), &
                       ior(ishft(int(s%buf(4*i - 1), int64), 8), int(s%buf(4*i), int64)))
        end do
        do i = 17, 64
            s0 = ieor(ieor(rotr(w(i - 15), 7), rotr(w(i - 15), 18)), ishft(w(i - 15), -3))
            s1 = ieor(ieor(rotr(w(i - 2), 17), rotr(w(i - 2), 19)), ishft(w(i - 2), -10))
            w(i) = iand(w(i - 16) + s0 + w(i - 7) + s1, MASK32)
        end do
        a = s%h(1); b = s%h(2); c = s%h(3); d = s%h(4)
        e = s%h(5); f = s%h(6); g = s%h(7); hh = s%h(8)
        do i = 1, 64
            s1 = ieor(ieor(rotr(e, 6), rotr(e, 11)), rotr(e, 25))
            t1 = iand(hh + s1 + ieor(iand(e, f), iand(ieor(e, MASK32), g)) + K(i) + w(i), MASK32)
            s0 = ieor(ieor(rotr(a, 2), rotr(a, 13)), rotr(a, 22))
            t2 = iand(s0 + ieor(ieor(iand(a, b), iand(a, c)), iand(b, c)), MASK32)
            hh = g; g = f; f = e
            e = iand(d + t1, MASK32)
            d = c; c = b; b = a
            a = iand(t1 + t2, MASK32)
        end do
        s%h(1) = iand(s%h(1) + a, MASK32); s%h(2) = iand(s%h(2) + b, MASK32)
        s%h(3) = iand(s%h(3) + c, MASK32); s%h(4) = iand(s%h(4) + d, MASK32)
        s%h(5) = iand(s%h(5) + e, MASK32); s%h(6) = iand(s%h(6) + f, MASK32)
        s%h(7) = iand(s%h(7) + g, MASK32); s%h(8) = iand(s%h(8) + hh, MASK32)
    end subroutine compress

    !> Bytes given as integers 0..255
    subroutine sha256_update_bytes(s, bytes)
        type(sha256_t), intent(inout) :: s
        integer, intent(in) :: bytes(:)
        integer :: i
        do i = 1, size(bytes)
            s%nbuf = s%nbuf + 1
            s%buf(s%nbuf) = iand(bytes(i), 255)
            if (s%nbuf == 64) then
                call compress(s)
                s%nbuf = 0
            end if
        end do
        s%nbytes = s%nbytes + size(bytes)
    end subroutine sha256_update_bytes

    !> An integer as 8 little-endian bytes
    subroutine sha256_update_int64(s, v)
        type(sha256_t), intent(inout) :: s
        integer(int64), intent(in) :: v
        integer :: b(8), i
        do i = 1, 8
            b(i) = int(iand(ishft(v, -8*(i - 1)), 255_int64))
        end do
        call sha256_update_bytes(s, b)
    end subroutine sha256_update_int64

    !> IEEE binary64 values as 8 little-endian bytes each (-0 is fed as +0)
    subroutine sha256_update_real64(s, x)
        type(sha256_t), intent(inout) :: s
        real(real64), intent(in) :: x(:)
        integer(int64) :: v
        integer :: i
        do i = 1, size(x)
            if (x(i) == 0.0_real64) then
                v = 0_int64
            else
                v = transfer(x(i), v)
            end if
            call sha256_update_int64(s, v)
        end do
    end subroutine sha256_update_real64

    subroutine sha256_update_string(s, str)
        type(sha256_t), intent(inout) :: s
        character(len=*), intent(in) :: str
        integer :: b(len(str)), i
        do i = 1, len(str)
            b(i) = iachar(str(i:i))
        end do
        call sha256_update_bytes(s, b)
    end subroutine sha256_update_string

    !> Finishes the digest (s is consumed) and returns it as 64 lower-case hex characters
    function sha256_hex(s) result(hex)
        type(sha256_t), intent(inout) :: s
        character(len=64) :: hex
        integer(int64) :: nbits
        integer :: i, j, pad(1), z(1), lenb(8)
        character(len=16), parameter :: DIGITS = '0123456789abcdef'
        integer(int64) :: word

        nbits = s%nbytes * 8_int64
        pad(1) = 128
        call sha256_update_bytes(s, pad)
        z(1) = 0
        do while (s%nbuf /= 56)
            call sha256_update_bytes(s, z)
        end do
        do i = 1, 8
            lenb(i) = int(iand(ishft(nbits, -8*(8 - i)), 255_int64))
        end do
        call sha256_update_bytes(s, lenb)
        do i = 1, 8
            word = s%h(i)
            do j = 1, 8
                hex(8*(i - 1) + j:8*(i - 1) + j) = DIGITS(int(iand(ishft(word, -4*(8 - j)), 15_int64)) + 1: &
                                                           int(iand(ishft(word, -4*(8 - j)), 15_int64)) + 1)
            end do
        end do
    end function sha256_hex

end module opensps_sha256
