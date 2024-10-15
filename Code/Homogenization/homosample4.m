%%
%Generate test sample
Jmin = 0.3;
Jmax = 3;
n = 8;
e = normr([1,1]);
niter = 20000;
data  = zeros(niter,n^2+1);


for i=1:niter
    i

    ax = Jmin + (Jmax-Jmin)*rand(n,n);
    ay = ax;
    A  = cell(n,n);
    axnew = zeros(n,n);
    aynew = zeros(n,n);
    for k=1:n
        for l=1:n
          axnew(k,l) = (ax(k,mod(l,n)+1) + ax(k,l))/2;
          aynew(k,l) = (ax(mod(k-2,n)+1,l) + ax(k,l))/2;
        end
    end

    [E1,u]  = conductance(e,axnew,aynew);

    data(i,1)     = E1;
    data(i,2:end) = (ax(:)-(Jmax+Jmin)/2)/((Jmax-Jmin)/sqrt(12));


end

dlmwrite(['homosample_n=' num2str(n) '.txt'],data)
