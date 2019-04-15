function [stats,statsz] = svg_get_sf( filename )
    [filepath, name, ext] = fileparts(filename);
    sff = strcat(filepath,"/" ,name, '.SF')
    sffz = strcat(filepath,"/" ,name, '.SFz')
    d = csvread(filename);
    dt = d';
    [stats, statsz] = svg_getFeatures(dt(:,:));
    statsz = statsz( :, any(statsz,1) );
    csvwrite(sff,stats);
    csvwrite(sffz,statsz);
    
