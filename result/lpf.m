
function y = lpf(input, prev_res, sampling_freq, cutoff_freq)
rc = 1.0/(cutoff_freq * 2*pi);
dt = 1.0/(sampling_freq);
a = dt/(rc+dt);

y= prev_res + a*(input-prev_res);
end