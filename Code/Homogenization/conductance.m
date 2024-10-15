function [E,u] = conductance(e,ax,ay)
n = size(ax,1);
L = zeros(n^2,n^2);
b = zeros(n^2,1);
e = e(:);
for i=1:n
  %Create Laplacian
  for j=1:n
    %Center
    L((i-1)*n+j,(i-1)*n+j)   = ax(i,j)+ax(i,mod(j-2,n)+1)+ay(mod(i,n)+1,j)+ay(i,j);

    %Right
    jj = mod(j,n)+1;
    L((i-1)*n+j,(i-1)*n+jj) = -ax(i,j);

    %Left
    jj = mod(j-2,n)+1;
    L((i-1)*n+j,(i-1)*n+jj) = - ax(i,jj);

    %Up
    ii = mod(i-2,n)+1;
    L((i-1)*n+j,(ii-1)*n+j) = -ay(i,j);

    %Down
    ii = mod(i,n)+1;
    L((i-1)*n+j,(ii-1)*n+j) = -ay(ii,j);


    %Create constant vectors
    jj = mod(j-2,n)+1;
    ii = mod(i,n)+1;
    b((i-1)*n+j) = e(1)*(ax(i,j) - ax(i,jj)) + e(2)*(ay(i,j) - ay(ii,j));

  end

end
h = 1/n;
L = L/h^2;
b = b/h;

u = L(2:n^2,2:n^2)\(-b(2:n^2));
u = [0;u];

%u = (L+ones(n^2))\(-b);

E = 1/n^2*(u'*L*u + 2*u'*b + sum(sum(ax+ay))/2);

% u = u-sum(u)/numel(u);
% norm(L*u+b)
% norm(u)
% E = 0;
% for i=1:n
%   for j=1:n
%     tmp = zeros(2,1);
%     tmp(1) = (u((i-1)*n+mod(j,n)+1) - u((i-1)*n+j))/h;
%     tmp(2) = (u(mod(i-2,n)*n+j) - u((i-1)*n+j))/h;
%     tmp    = tmp+e;
%     E      = E+(tmp'*diag([ax(i,j);ay(i,j)])*tmp)/n^2;
%   end
% end
