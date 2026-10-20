module fsps_special_functions
    !> @brief
    !> Provides mathematical special functions needed for FSPS physics.
    
    use fsps_precision, only: WP
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan, ieee_negative_inf
    implicit none

    private

    public :: mag_from_flux
    public :: expi
    public :: gammainc

    ! ------------------------------------------------------------------------
    ! CONSTANTS
    ! ------------------------------------------------------------------------

    ! Euler-Mascheroni constant
    real(WP), parameter :: GAMMA = 0.57721566490153286060651209008240243104215933593992_wp
    real(WP), parameter :: EPS = 1.0e-20_wp
    real(WP), parameter :: LOG10_FACTOR = 2.5_wp

contains

    !> @brief
    !> Computes the astronomical magnitude from a given flux.
    !>
    !> @details
    !> Returns NaN if flux <= 0.
    !>
    !> @param[in] flux The input flux.
    !> @return    The magnitude, or NaN if undefined.
    elemental function mag_from_flux(flux) result(mag)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: flux
        real(WP) :: mag

        ! Branchless-friendly logic (better for AD/GPU)
        if (flux > 0.0_wp) then
            mag = -LOG10_FACTOR * log10(flux)
        else
            ! Return a signal, don't crash the car
            mag = get_quiet_nan()
        end if
    end function mag_from_flux

    !> @brief
    !> Computes the Exponential Integral Ei(x).
    !>
    !> @details
    !> Based on the implementation by Shanjie Zhang and Jianming Jin.
    !> Uses a power series expansion for small x (|x| <= 40) and an
    !> asymptotic expansion for large x.
    !>
    !> @param[in] x  The argument (must be > 0 for real result).
    !> @return    res The value of Ei(x). 
    !>                Returns -Infinity at x=0.
    !>                Returns NaN for x < 0.
    elemental function expi(x) result(res)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: x
        real(WP) :: res

        integer :: k
        integer, parameter :: MAXIT = 1000
        real(WP) :: r, inv_x, term_denom
        
        if (x == 0.0_wp) then
            ! Return negative infinity
            res = get_neg_infinity()
            return
        else if (x < 0.0_wp) then
            ! Ei(x) is complex for x < 0; return NaN in this real implementation
            res = get_quiet_nan()
            return
        end if

        if (abs(x) <= 40.0_wp) then
            ! Power series expansion around x=0
            ! Ei(x) = Gamma + ln(x) + sum(x^k / (k * k!))
            res = 1.0_wp
            r = 1.0_wp
            
            do k = 1, MAXIT
                term_denom = real(k, WP) + 1.0_wp
                r = (r * real(k, WP) * x) / (term_denom * term_denom)

                res = res + r
                
                ! Convergence check
                if (abs(r) <= EPS * abs(res)) exit
                
                ! Check for non-convergence
                if (k == MAXIT) then
                    res = get_quiet_nan()
                    return
                end if
            end do
            
            res = GAMMA + log(x) + x * res

        else
            ! Asymptotic expansion for large x (divergent series, limited terms)
            ! Ei(x) ~ exp(x)/x * (1 + 1!/x + 2!/x^2 + ...)
            res = 1.0_wp
            r = 1.0_wp
            inv_x = 1.0_wp / x
            
            do k = 1, 20
                r = r * real(k, WP) * inv_x
                res = res + r
            end do
            
            res = exp(x) * inv_x * res
        end if

    end function expi

    !> @brief
    !> Computes the Regularized Lower Incomplete Gamma Function P(a, x) for integer 'a'.
    !>
    !> @details
    !> The regularized lower incomplete gamma function is defined as:
    !> P(a, x) = (1 / Gamma(a)) * Integral(t^(a-1) * exp(-t) dt) from 0 to x.
    !>
    !> For integer a, this has the closed form:
    !> P(a, x) = 1 - exp(-x) * Sum(x^k / k!) for k=0 to a-1.
    !>
    !> This implementation includes specific optimizations for a=1 (Exponential CDF)
    !> and a=2 to avoid loop overhead and minimize floating point operations.
    !> It uses a helper function (fsps_expm1) to preserve numerical precision 
    !> when x is very small.
    !>
    !> @param[in] power  The shape parameter 'a'. Must be a positive integer >= 1.
    !> @param[in] arg    The upper limit of integration 'x'. Must be >= 0.
    !> @return    res    The value of P(a, x). Returns 0.0 if arg < 0.
    elemental function gammainc(power, arg) result(res)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        integer, intent(in) :: power
        real(WP), intent(in) :: arg
        real(WP) :: res
        
        real(WP) :: sum_term, term, prefactor, real_power, log_fact
        integer :: k

        if (arg < 0.0_WP) then
            res = 0.0_WP
            return
        endif

        if (power == 1) then
            ! P(1, x) = 1 - e^(-x)
            ! Use helper to avoid cancellation when arg is small
            res = -fsps_expm1(-arg)
            
        else if (power == 2) then
            ! P(2, x) = 1 - e^(-x) - x*e^(-x)
            !         = (1 - e^(-x)) - x*e^(-x)
            !         = -expm1(-arg) - arg * exp(-arg)
            res = -fsps_expm1(-arg) - arg * exp(-arg)
            
        else
            ! General case:
            ! Use a numerically stable branch when arg < power (small lower tail),
            ! and use the finite-sum complement form otherwise.
            real_power = real(power, WP)

            if (arg == 0.0_wp) then
                res = 0.0_wp
            else if (arg < real_power) then
                ! P(a,x) = exp(-x) * x^a / Gamma(a+1) * sum_{n=0..inf} x^n / ((a+1)...(a+n))
                sum_term = 1.0_wp
                term = 1.0_wp

                do k = 1, 10000
                    term = term * (arg / (real_power + real(k, WP)))
                    sum_term = sum_term + term
                    if (abs(term) <= 10.0_wp * epsilon(1.0_wp) * abs(sum_term)) exit
                end do

                ! log(power!) as a sum (power is an integer; avoids log_gamma, which has no GPU libm on AMD)
                log_fact = 0.0_wp
                do k = 2, power
                    log_fact = log_fact + log(real(k, WP))
                end do
                prefactor = exp(real_power * log(arg) - arg - log_fact)
                res = prefactor * sum_term
            else
                sum_term = 1.0_WP
                term = 1.0_WP

                do k = 1, power - 1
                    term = term * (arg / real(k, WP))
                    sum_term = sum_term + term
                end do

                res = 1.0_WP - exp(-arg) * sum_term
            end if
        endif

        if (res < 0.0_wp) res = 0.0_wp
        if (res > 1.0_wp) res = 1.0_wp

    end function gammainc

    ! ------------------------------------------------------------------------
    ! PRIVATE HELPER ROUTINES
    ! ------------------------------------------------------------------------

    !> @brief Portable implementation of exp(x) - 1
    !> Handles small x via Taylor series to avoid precision loss.
    elemental function fsps_expm1(x) result(val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP), intent(in) :: x
        real(WP) :: val
        
        ! Threshold: if |x| < 1e-5, standard exp(x)-1 loses bits
        ! but Taylor series converges very rapidly.
        if (abs(x) < 1.0e-5_wp) then
            ! Series: x + x^2/2! + x^3/3! + ...
            val = x * (1.0_wp + x * (0.5_wp + x * (1.0_wp/6.0_wp)))
        else
            val = exp(x) - 1.0_wp
        end if
    end function fsps_expm1

    !> @brief Helper to generate a Quiet NaN
    pure function get_quiet_nan() result(val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP) :: val
        val = ieee_value(0.0_wp, ieee_quiet_nan)
    end function get_quiet_nan

    !> @brief Helper to generate Negative Infinity
    pure function get_neg_infinity() result(val)
        !$acc routine seq
#ifdef OMP_TARGET
        !$omp declare target
#endif
        real(WP) :: val
        val = ieee_value(0.0_wp, ieee_negative_inf)
    end function get_neg_infinity

end module fsps_special_functions