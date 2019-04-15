function [stats] = svg_get_sf( filename )
    [filepath, name, ext] = fileparts(filename);
    sff = strcat(filepath,"/" ,name, '.SF')
    d = csvread(filename);
    dt = d';
    [stats, nop] = svg_getFeatures(dt(:,:));
    csvwrite(sff,stats);
    
