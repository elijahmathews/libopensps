# agn_dust.f90: upstream d9a387a (cconroy20/fsps#81) scales the AGN torus attenuation by dust2.
s|agnspeci = agnspeci\*EXP(-attn_curve(spec_lambda,dust_type,pset))|agnspeci = agnspeci*EXP(-pset%dust2 * attn_curve(spec_lambda,dust_type,pset))|
