%Skeleton Script for implementing exercise 1 with Reaction Band.
%This script needs to be kept in the same folder as the other scripts e.g.
%'Client.m' etc. which comes as part of the Vicon Datastream SDK for it to function properly.

%SUPER DUPER IMPORTANT: There two dependencies which need to initialised before this script will
%run.
%1. Switch on the Reaction Band, it will automatically start broadcasting as a
%Bluetooth low energy device.
%2. Open the application 'Bluetooth Console App.exe', this will make the
%initial connection to Reaction Band. If you do not see the message
%'Waiting for TCP connection...' after 10 seconds, close the application
%and repeat steps 1 and 2.
%3. Have fun experimenting :)
clear, clc

Start = false;

Target = 250;

Tolerance =5;

%Array we will store Xcoordinates for experiment
Xcoordinate = [];

%Array for storing whether or not in target area
TargetArea = [];

% Program options
TransmitMulticast = false;
EnableReaction1 = true;
bReadCentroids = false;
bReadRays = false;
axisMapping = 'ZUp';

% A dialog to stop the loop
MessageBox = msgbox( 'Stop DataStream Client', 'Vicon DataStream SDK' );

%Construct an instance of Reaction
Reaction1 = Reaction('127.0.0.1',13000);
Reaction1.Connect();

% Load the Vicon SDK
fprintf( 'Loading SDK...' );
Client.LoadViconDataStreamSDK();
fprintf( 'done\n' );

% Program options
HostName = 'localhost:801';

% Make a new client
MyClient = Client();

% Connect to a server
fprintf( 'Connecting to %s ...', HostName );
while ~MyClient.IsConnected().Connected
  % Direct connection
  MyClient.Connect( HostName );
  
  % Multicast connection
  % MyClient.ConnectToMulticast( HostName, '224.0.0.0' );
  
  fprintf( '.' );
end
fprintf( '\n' );

% Enable some different data types
MyClient.EnableSegmentData();
MyClient.EnableMarkerData();
MyClient.EnableUnlabeledMarkerData();
MyClient.EnableDeviceData();
if bReadCentroids
  MyClient.EnableCentroidData();
end
if bReadRays
  MyClient.EnableMarkerRayData();
end

%fprintf( 'Segment Data Enabled: %s\n',          AdaptBool( MyClient.IsSegmentDataEnabled().Enabled ) );
fprintf( 'Marker Data Enabled: %s\n',           AdaptBool( MyClient.IsMarkerDataEnabled().Enabled ) );
fprintf( 'Unlabeled Marker Data Enabled: %s\n', AdaptBool( MyClient.IsUnlabeledMarkerDataEnabled().Enabled ) );
%fprintf( 'Device Data Enabled: %s\n',           AdaptBool( MyClient.IsDeviceDataEnabled().Enabled ) );
% fprintf( 'Centroid Data Enabled: %s\n',         AdaptBool( MyClient.IsCentroidDataEnabled().Enabled ) );
% fprintf( 'Marker Ray Data Enabled: %s\n',       AdaptBool( MyClient.IsMarkerRayDataEnabled().Enabled ) );

% Set the streaming mode
MyClient.SetStreamMode( StreamMode.ClientPull );
% MyClient.SetStreamMode( StreamMode.ClientPullPreFetch );
% MyClient.SetStreamMode( StreamMode.ServerPush );

% Set the global up axis
if axisMapping == 'XUp'
  MyClient.SetAxisMapping( Direction.Up, ...
                          Direction.Forward,      ...
                          Direction.Left ); % X-up
elseif axisMapping == 'YUp'
  MyClient.SetAxisMapping( Direction.Forward, ...
                         Direction.Up,    ...
                         Direction.Right );    % Y-up
else
  MyClient.SetAxisMapping( Direction.Forward, ...
                         Direction.Left,    ...
                         Direction.Up );    % Z-up
end

Output_GetAxisMapping = MyClient.GetAxisMapping();
fprintf( 'Axis Mapping: X-%s Y-%s Z-%s\n', Output_GetAxisMapping.XAxis.ToString(), ...
                                           Output_GetAxisMapping.YAxis.ToString(), ...
                                           Output_GetAxisMapping.ZAxis.ToString() );


% Discover the version number
Output_GetVersion = MyClient.GetVersion();
fprintf( 'Version: %d.%d.%d\n', Output_GetVersion.Major, ...
                                Output_GetVersion.Minor, ...
                                Output_GetVersion.Point );
  
if TransmitMulticast
  MyClient.StartTransmittingMulticast( 'localhost', '224.0.0.0' );
end  

Counter = 1;

%Main Program While Loop
% Loop until the message box is dismissed

while ishandle( MessageBox )
  drawnow;
  Counter = Counter + 1;
  
  % Get a frame
  fprintf( 'Waiting for new frame...' );
  while MyClient.GetFrame().Result.Value ~= Result.Success
    fprintf( '.' );
  end% while
  fprintf( '\n' );  

  % Get the unlabeled markers
  UnlabeledMarkerCount = MyClient.GetUnlabeledMarkerCount().MarkerCount;
  fprintf( '    Unlabeled Markers (%d):\n', UnlabeledMarkerCount );
  for UnlabeledMarkerIndex = 1:UnlabeledMarkerCount
    % Get the global marker translation
    Output_GetUnlabeledMarkerGlobalTranslation = MyClient.GetUnlabeledMarkerGlobalTranslation( UnlabeledMarkerIndex );
    
    %Only look at unlabeled markers within 1m radius of origin
    if (abs(Output_GetUnlabeledMarkerGlobalTranslation(1)) < 1000 && abs(Output_GetUnlabeledMarkerGlobalTranslation(2)) < 1000)
        
        new_X = Output_GetUnlabeledMarkerGlobalTranslation(1);
        
        fprintf( '      Marker #%d: (%g, %g, %g)\n',                                    ...
                       UnlabeledMarkerIndex - 1,                                    ...
                       Output_GetUnlabeledMarkerGlobalTranslation.Translation( 1 ), ...
                       Output_GetUnlabeledMarkerGlobalTranslation.Translation( 2 ), ...
                       Output_GetUnlabeledMarkerGlobalTranslation.Translation( 3 ) );
        
        %Give test feedback when X coordinate of marker is > 20 from origin
        if (Xcoordinate > 100)
          Reaction1.motorON();
        else
          Reaction1.motorOFF();
        end
    
        %Starting experiment
        while (Start)

          if (Xcoordinate > Target)
              Reaction1.motorON();
          else
              Reaction1.motorOFF();
          end

          %Datalog Xcoordinate
          Xcoordinate = [Xcoordinate, new_X]

          if ((Target - new_X)< 5)
              TargetArea = [TargetArea, 1];
          else 
              TargetArea = [TargetArea, 0];
          end

          %If person is able to stay within target area for > 2seconds
          if (length(TargetArea)>200)
              %If last 200 have been 1's then finish
              if (min(TargetArea((length(TargetArea)+1-200):(length(TargetArea))))>0)
                  Start = false;
                  Reaction1.stopArduino();
                  %FinalFrame = MyClient.GetFrameNumber();
              end
          end
          
        end
    end
        
  end



end 


%Disconnect the Reaction Band
Reaction1.Disconnect();


if TransmitMulticast
  MyClient.StopTransmittingMulticast();
end  

% Disconnect and dispose
MyClient.Disconnect();

% Unload the SDK
fprintf( 'Unloading SDK...' );
Client.UnloadViconDataStreamSDK();
fprintf( 'done\n' );


