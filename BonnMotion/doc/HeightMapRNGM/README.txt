#
# BonnMotion - a mobility scenario generation and analysis tool             
# Copyright (C) 2018--2020 Perspecta Labs Inc.
#                                                                           
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# This work was supported by the Defense Advanced Research Projects Agency
# (DARPA) under Contract No. HR0011-17-C-0047. Any opinions, findings,
# conclusions or recommendations expressed in this material are those of
# the authors and do not necessarily reflect the views of DARPA.
#
# DISTRIBUTION STATEMENT A. Approved for public release.

This file describes the use and implementation of the Height Map Reference Node
Group Mobility (HM-RNGM) model for the BonnMotion generator.  The HM-RNGM
model is based on the Reference Point Group Mobility Model (RPGM).  HM-RNGM
enhances RPGM by adding support for limited three-dimensional mobility by
calculating the height of the node at its new location and calculating node
speed based on the 3D Euclidean distance to that point.  In addition, HM-RNGM
allows node groups to be statically defined in a configuration file.  Finally,
HM-RNGM allows the statically defined group to follow a node as its reference
point rather than creating an abstract point that the nodes follow.

Installation
------------
The HM-RNGM is installed with the other BonnMotion classes.  However, it
requires that the gdal library be available in the standard library path or in
/usr/gdal2/lib.

Invoking HM-RNGM
----------------

HM-RNGM is invoked through the standard BonnMotion BM class with the model
name HeightMapRNGM.  Note that the model must be invoked with the '-J 3D'
command-line option.  The model takes the following parameters in addition to
those defined by the RandomSpeedBase model:

-e  The number of subsegments each following node in a group takes for each
    segment of the reference node's motion.

-g  The path to the membership group file.

-m  The multiple of the maximum speed at which the following nodes can move
    during subsegment motion.

-o  Geographic position (latitude, longitude) to use as the origin rather than
    the natural origin of the terrain map (position 0,0 in the terrain file's
    raster).  The format of the position is ISO6709 annex H.  (E.g.
    +010203.45-0060708.9 for 1* 2' 3.45" N 6* 7' 8.9" W)

-r  The maximum distance from group leader to each member

-t  Path to the terrain file that defines the height of the terrain at each
    location.  The file must be readable by the GDAL library and must define a
    spatial reference system that is convertable to WGS84.  The file must
    report height (values) in meters.  HM-RNGM will create mobility files
    whose x and y coordinates are meter offsets from the origin of the terrain
    file in the direction of the terrain file's projection.


Configuring Membership Group File
---------------------------------------

The default membership group file is "node_groups.txt". This default file is in
the format defined by Java Properties.  The requires key is groups, which
contains a comma-separated list of names of groups.  Each group has two
possible keys: nodes_<groupname> and bounding_box_<groupname>.  The value
associated with the nodes_<> key is a list of comma-separated numeric ranges
defining the node IDs for nodes that are part of the groups.  The value
associated with the bounding_box_<groupname> defines the lower left and upper
right corners of the box in which the nodes of the groups may move.  The
specification is either 4 space-separated integers or two space-separated
ISO6709 Annex H positions.  In addition, nodes may specified a position using
the key position_node_<nodeid> for which the value is either 2 (or 3 to
indicated height above ground) space separated integers for an offset in meters
from 0 0 or a single ISO6709 Annex H position.
