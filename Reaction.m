% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) SAGE Motion Systems 2018.
% All rights reserved.  This software is protected by copyright
% law and international treaties.  No part of this software / document
% may be reproduced or distributed in any form or by any means,
% whether transiently or incidentally to some other use of this software,
% without the written permission of the copyright owner.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part of the Reaction SDK for MATLAB.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Reaction
    properties % (SetAccess = private)
        object
    end
    methods (Static = true)
        
    end
    methods
        %%Constructor
        function obj = Reaction(ipaddress, port)
            obj.object = tcpip(ipaddress, port);
        end            
        %%Connect to Reaction Bluetooth Console application
        function Connect( obj )
            fopen(obj.object);
        end
        %%Disconnect Reaction from Bluetooth Console application
        function Disconnect(obj)
            fwrite(obj.object, 97, 'uint8');
            fwrite(obj.object, 88, 'uint8');
            %fwrite(obj.object, 0, 'uint8');
            fclose(obj.object);
        end
        %%Turn motor On
        function motorON(obj)
            fwrite(obj.object, 98, 'uint8');
            fwrite(obj.object, 88, 'uint8');
        end
        %%Turn motor Off
        function motorOFF(obj)
            fwrite(obj.object, 97, 'uint8');
            fwrite(obj.object, 88, 'uint8');
        end
        %%Stop current command
        function stopArduino(obj)
            fwrite(obj.object, 0, 'uint8');
        end
    end
end
