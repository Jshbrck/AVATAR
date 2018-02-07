function obj = readObj(fname)
    %
    % obj = readObj(fname)
    % This function can read styku or fit3d_proscanner file
    %
    % INPUT: fname - wavefront object file full path
    %
    % OUTPUT: obj.v - mesh vertices
    %       : obj.f - face definition assuming faces are made of of 3 vertices
    
    % Find number of headerlines
    fileID = fopen(fname);
    headerLines = 0;
    while 1
        tline = fgetl(fileID);
        ln = sscanf(tline,'%s',1); % line type
        if strcmp(ln,'v') || strcmp(ln,'f')
            break 
        end
        headerLines = headerLines + 1;
    end
    fclose(fileID);
    
    % Get v and f
    fileID = fopen(fname);
    C = textscan(fileID,'%s %f %f %f','HeaderLines', headerLines);
    fclose(fileID);

    id = char(C{1});
    vn = nnz(id == 'v'); % number of vertices
    fn = nnz(id == 'f'); % number of faces
    
    v = zeros(vn,3);
    v(:,1) = C{2}(id == 'v');
    v(:,2) = C{3}(id == 'v');
    v(:,3) = C{4}(id == 'v');

    f = zeros(fn,3);
    f(:,1) = C{2}(id == 'f');
    f(:,2) = C{3}(id == 'f');
    f(:,3) = C{4}(id == 'f');
    
    % set up matlab object 
    obj.v = v; obj.f = f;
end