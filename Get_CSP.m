function V = Get_CSP(X_left,X_right,filters,csp_corr,csp_diff)

if csp_diff
    X_left = X_left(2:end,:) - X_left(1:end-1,:);
    X_right = X_right(2:end,:) - X_right(1:end-1,:);
end

n_CSPs = size(X_right,2)/16; % FBCSP if > 1

V = {};

for i = 1:n_CSPs
    
    if csp_corr
        cov_left = corr(X_left(:,16*(i-1)+1:16*i));
        cov_right = corr(X_right(:,16*(i-1)+1:16*i));
    else
        cov_left = cov(X_left(:,16*(i-1)+1:16*i));
        cov_right = cov(X_right(:,16*(i-1)+1:16*i));
    end



    [v,l] = eig(cov_left,cov_left+cov_right);

    [~,idcs] = sort(diag(l));
    v = v(:,idcs);
    v = v(:,[1:filters/2,end-filters/2+1:end]);

    V{i} = v;

end


end