function r = g (x)
  r = [x(1) + x(2) + x(3) + x(4) + x(5) - 1];
endfunction

function obj = fn(x)
	lam = [0.61229313169297480 0.8437346821964328 5.033100125970339 2.4658403339555597 3.1037769659195202];
	thet = [-0.03371212188158611 -0.046502704806702023 -0.2776997643101508 -0.13602926735116322 -0.17124393186657508];
	sigma = 0.32;

	obj=0;
	for i = 1:5
		obj = obj - log2(lam(i)*sigma*sqrt(x(i)) + thet(i));
	endfor
endfunction

x0 = [0.30; 0.15; 0.18; 0.20; 0.17];
[x, obj, info, iter, nf, lambda] = sqp (x0, @fn, @g, [], 0, +realmax)
