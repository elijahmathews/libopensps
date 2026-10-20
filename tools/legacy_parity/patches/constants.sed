# sps_vars.f90: replaces legacy's rounded constants by the fork's (IAU 2015 / CODATA 2018 and the
# exact 1/(4 pi sigma)), so that the matrix compares code, not constants (cconroy20/fsps#83).
s|gsig4pi = 1/4.13E10|gsig4pi = 1/4.0479831348E10_sp|
s|mypi    = 3.14159265$|mypi    = 3.14159265358979_sp|
s|hck     = 1.43878E8|hck     = 1.438777E8_sp|
s|clight  = 2.9979E18|clight  = 2.99792458E18_sp|
s|hc2sig  = 0.105021|hc2sig  = 0.105023_sp|
s|msun    = 1.989E33|msun    = 1.988410E33_sp|
s|rsun    = 6.955E10|rsun    = 6.957E10_sp|
s|lsun    = 3.839E33|lsun    = 3.828E33_sp|
s|newton  = 6.67428E-8|newton  = 6.6743E-8_sp|
s|pc2cm   = 3.08568E18|pc2cm   = 3.0856775814913673E18_sp|
s|yr2sc   = 3.15569E7|yr2sc   = 3.15576E7_sp|
s|hplank  = 6.6261E-27|hplank  = 6.62607015E-27_sp|
