#!/usr/bin/octave -qf

file=fopen('metadata.dat','wt');
for a=1:33
	X=['./position.',num2str(a),'.dat ',num2str(-20.35+0.35*(a-1)),' 8'];
	fprintf(file,X);
	fprintf(file,'\n');
end
