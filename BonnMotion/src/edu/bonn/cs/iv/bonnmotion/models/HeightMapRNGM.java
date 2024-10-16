/*******************************************************************************
 ** BonnMotion - a mobility scenario generation and analysis tool             **
 ** Copyright (C) 2018--2019 Perspecta Labs Inc.                              **
 **                                                                           **
 ** This program is free software; you can redistribute it and/or modify      **
 ** it under the terms of the GNU General Public License as published by      **
 ** the Free Software Foundation; either version 2 of the License, or         **
 ** (at your option) any later version.                                       **
 **                                                                           **
 ** This program is distributed in the hope that it will be useful,           **
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of            **
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             **
 ** GNU General Public License for more details.                              **
 **                                                                           **
 ** You should have received a copy of the GNU General Public License         **
 ** along with this program; if not, write to the Free Software               **
 ** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA **
 **                                                                           **
 ** This work was supported by the Defense Advanced Research Projects Agency  **
 ** (DARPA) under Contract No. HR0011-17-C-0047. Any opinions, findings,      **
 ** conclusions or recommendations expressed in this material are those of    **
 ** the authors and do not necessarily reflect the views of DARPA.            **
 **                                                                           **
 ** DISTRIBUTION STATEMENT A. Approved for public release.                    **
 *******************************************************************************/

package edu.bonn.cs.iv.bonnmotion.models;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import com.perspectalabs.bonnmotion.util.HeightMap;
import com.perspectalabs.bonnmotion.util.PositionGeoParser;
import com.perspectalabs.bonnmotion.util.StationaryNode;

import edu.bonn.cs.iv.bonnmotion.*;
import edu.bonn.cs.iv.bonnmotion.printer.Dimension;
import edu.bonn.cs.iv.util.maps.PositionGeo;

/**
 * Height Map Reference Node Group Mobility
 *
 * A mobility generation module based on RPGM {@link RPGM}. Modified to:
 *
 * <ul>
 * <li>add support for using a height map to set the height of nodes</li>
 * <li>to set the reference point of a group to be a node rather than an
 * abstract point</li>
 * <li>to define groups statically</li>
 * <li>to allow groups to be restricted to operating within a particular
 * bounding box</li>
 * </ul>
 *
 * @author Yitzchak M. Gottlieb <ygottlieb@perspectalabs.com>
 * @author Matthew Witkowsi <mwitkowski@perspectalabs.com>
 */

public class HeightMapRNGM extends RandomSpeedBase {
    private static ModuleInfo info;

    static {
        info = new ModuleInfo("HeightMapRNGM");
        info.description = "Application to create movement scenarios according to the Reference Node Group Mobility model with support for 3D motion and stationary nodes";

        info.major = 1;
        info.minor = 0;
        info.revision = 0;

        info.contacts.add("Alex Poylisher <apoylisher@perspectalabs.com>");
        info.contacts.add("Yitzchak M. Gottlieb <ygottlieb@perspectalabs.com>");
        info.contacts.add("Matthew Witkowski <mwitkowski@perspectalabs.com>");
        info.authors.add("Perspecta Labs Inc.");
        info.affiliation = "Perspecta Labs Inc. <https://www.perspectalabs.com>";
    }

    public static ModuleInfo getInfo() {
        return info;
    }

    /**
     * This class encapulates the information about a group of nodes. The group
     * has the following properties:
     * <ul>
     * <li>A name {@link #getName}</li>
     * <li>A group leader {@link #getLeader}</li>
     * <li>A collection of member nodes {@link #getNodes}</li>
     * <li>A bounding box in the meters from 0,0 space consisting of a lower
     * left {@link #getLl} corner and an upper right {@link #getUr} corner</li>
     * </ul>
     * 
     * @author ygottlieb
     *
     */
    private static class NodeGroup {

        /** The name of the group */
        private String name;

        /** The IDs of the members of the group */
        private List<Integer> nodes = new ArrayList<>();

        /** The lower left corner of the group;s bounding box. */
        private Position ll = new Position(0, 0);

        /** The upper right corner of the group's bounding box. */
        private Position ur = new Position(0, 0);

        /** The node ID of the group's leader/reference node. */
        private int leader;

        /**
         * Create a new node group
         * 
         * @param name
         *            The name of the group
         * @param leader
         *            The node ID of the group's leader
         * @raises IllegalArgumentException if the leader's ID is negative
         */
        public NodeGroup(String name, int leader) {
            this.name = name;

            if (leader < 0) {
                throw new IllegalArgumentException("Group " + name
                        + "'s leader node has a negative index: " + leader);
            } else {
                this.leader = leader;
            }
        }

        /**
         * Create a node group with an explicit bounding box
         * 
         * @param name
         *            The name of the group
         * @param leader
         *            The node ID of the group's leader
         * @param ll
         *            The lower left corner of the group's bounding box
         * @param ur
         *            The upper right corner of the group's bounding box
         * @raises IllegalArgumentException if the leader's ID is negative or
         *         the bounding box is misordered.
         */
        public NodeGroup(String name, int leader, Position ll, Position ur) {
            this(name, leader);

            setBoundingBox(ll, ur);
        }

        /**
         * Create a node group with an implicit bounding box from (0,0) to (x,y)
         * 
         * @param name
         *            The name of the group
         * @param leader
         *            The node ID of the group's leader
         * @param x
         *            The x coordinate of the upper right corner of the bounding
         *            box
         * @param y
         *            The y coordinate of the upper right corner of the bounding
         *            box
         */
        public NodeGroup(String name, int leader, double x, double y) {
            this(name, leader, new Position(0, 0), new Position(x, y));
        }

        /**
         * Add a node to the group
         * 
         * @param node
         *            The ID of the node to add
         * @raise IllegalArgumentException if node is negative or the leader's
         *        ID
         */
        public void addNode(int node) {
            addNodes(Collections.singleton(node));
        }

        /**
         * Add all the nodes to the group.
         * 
         * @param nodes
         *            The nodes to add
         * 
         * @raise IllegalArgumentException if any of node IDs are negative,
         *        leader's ID, or already in group
         * 
         */
        public void addNodes(Collection<Integer> nodes) {

            Stream<Integer> negativeIDs = nodes.stream().filter(i -> i < 0);

            Set<Integer> intersection = new HashSet<>(nodes);
            intersection.retainAll(this.nodes);

            if (negativeIDs.count() > 0) {
                throw new IllegalArgumentException("Nodes have negative ID: "
                        + negativeIDs.sorted().toArray());
            } else if (nodes.stream().filter(i -> i == leader).count() > 0) {
                throw new IllegalArgumentException(
                        "Node ID is leader ID: " + leader);
            } else if (intersection.size() > 0) {
                throw new IllegalArgumentException(
                        "Nodes are already in the group: " + intersection);
            } else {
                this.nodes.addAll(nodes);
            }
        }

        /**
         * Set the bounding box of the group. The format of the string is
         * either:
         * 
         * <ol>
         * <li>four, space-separated doubles:
         * <ul>
         * <li>lower left X</li>
         * <li>lower left Y</li>
         * <li>upper right X</li>
         * <li>upper right Y</li>
         * </ul>
         * or</li>
         * <li>a space-separated pair of ISO6709 coordinates</li>
         * </ol>
         * 
         * 
         * @param boundingBox
         *            A string formatted as above representing the bounding box
         * @param heightMap
         *            The height map of the model
         * @raises IllegalArgumentException see
         *         {@link #setBoundingBox(Position, Position)}
         * @raises IllegalArgumentException if the bounding box has coordinates
         *         and heightMap is null
         */
        public void setBoundingBox(String boundingBox, HeightMap heightMap) {
            String[] parts = LIST_SEPARATOR.split(boundingBox);

            switch (parts.length) {
            case 2: {
                if (heightMap == null) {
                    throw new IllegalArgumentException("Group " + getName()
                            + ": Cannot parse bounding box " + boundingBox
                            + " without a heightMap");
                } else {
                    Position ll = heightMap.transformFromWgs84ToPosition(
                            PositionGeoParser.parsePositionGeo(parts[0]));
                    Position ur = heightMap.transformFromWgs84ToPosition(
                            PositionGeoParser.parsePositionGeo(parts[1]));

                    try {
                        setBoundingBox(ll, ur);
                    } catch (IllegalArgumentException e) {
                        throw new IllegalArgumentException("Group " + getName()
                                + ": Cannot set bounding box " + boundingBox,
                                e);
                    }
                }

            }
                break;
            case 4: {
                setBoundingBox(
                        new Position(Double.parseDouble(parts[0]),
                                Double.parseDouble(parts[1])),
                        new Position(Double.parseDouble(parts[2]),
                                Double.parseDouble(parts[3])));

            }
                break;

            default:
                throw new IllegalArgumentException("Group " + getName()
                        + ": Cannot parse bounding box " + boundingBox);
            }
        }

        /**
         * Set the bounding box of the group
         * 
         * @param ll
         *            The lower left corner
         * @param ur
         *            The upper right corner
         * @raises IllegalArgumentException if the lower left corner is above or
         *         to the right of the upper right corner.
         */
        public void setBoundingBox(Position ll, Position ur) {

            if (ll.x > ur.x || ll.y > ur.y) {
                throw new IllegalArgumentException("Group " + getName()
                        + ": The lower left corner " + ll
                        + " is not below and to the left of the upper right corner "
                        + ur);
            } else {
                this.ll = new Position(ll);
                this.ur = new Position(ur);
            }
        }

        /**
         * @return the name of the group
         */
        public String getName() {
            return name;
        }

        /**
         * @return the node ID of the leader/reference node of the group
         */
        public int getLeader() {
            return leader;
        }

        /**
         * @return the node IDs of the members of the group
         */
        public List<Integer> getNodes() {
            return Collections.unmodifiableList(nodes);
        }

        /**
         * @return the lower left corner of the bounding box
         */
        public Position getLl() {
            return ll;
        }

        /**
         * @return the upper right corner of the bounding box
         */
        public Position getUr() {
            return ur;
        }

        /** The key in the groups file giving the name of groups */
        public static final String GROUPS_KEY = "groups";

        /**
         * The prefix of keys in the groups file giving IDs of group members of
         * named groups
         */
        public static final String GROUP_NODES_KEY_PREFIX = "nodes_";

        /**
         * The prefix of keys in the groups file giving the bounding box of
         * named groups
         */
        public static final String GROUP_BOUNDING_BOX_KEY_PREFIX = "bounding_box_";

        /** List separators for group and node lists: comma or spaces */
        public static final Pattern LIST_SEPARATOR = Pattern
                .compile("(,\\s*)|\\s+");

        /** Node lists may include ID ranges */
        public static final Pattern NODE_RANGE = Pattern
                .compile("(\\d+)(-(\\d+))?");

        /**
         * Parse the string representation of the node list. The list is a
         * space- or comma-separated list of integers or ranges of integers.
         * 
         * @param nodelist
         *            The string representation of a list of node IDs
         * @return A Collection of node IDs, empty if nodelist is null or the
         *         empty string
         * @raises IllegalArgumentException if any elements of the list are not
         *         integers or ranges
         */
        private static List<Integer> parseNodeList(String nodelist) {
            List<Integer> retval = new ArrayList<>();

            if (nodelist != null && !nodelist.isEmpty()) {
                for (String part : LIST_SEPARATOR.split(nodelist)) {
                    Matcher match = NODE_RANGE.matcher(part);
                    if (!match.matches()) {
                        throw new IllegalArgumentException("Node list element '"
                                + part + "' is not an integer or range.");
                    } else {
                        int low = Integer.parseInt(match.group(1));
                        int high;

                        try {
                            high = Integer.parseInt(match.group(3));
                        } catch (NumberFormatException e) {
                            // The NODE_RANGE regular expression ensures that
                            // the match group is all digits, so NFE is only if
                            // the match group is null.
                            high = low;
                        }

                        for (int i = low; i <= high; ++i) {
                            retval.add(i);
                        }
                    }
                }
            }

            return retval;
        }

        /**
         * Parse the configuration file to return all the groups defined it.
         * 
         * @param configfile
         *            The contents of the configuration files
         * @param x
         *            The X coordinate of the upper right corner of the overall
         *            bounding box
         * @param y
         *            The Y coordinate of the upper right corner of the overall
         *            bounding box
         * @return A map between the name of a group and the NodeGroup object
         *         representing it.
         * @raises IllegalArgumentException if any of the group member
         *         definitions is invalid or empty
         * @RuntimeException if there are duplicate group names
         */
        public static Map<String, NodeGroup> parseGroups(Properties configfile,
                HeightMapRNGM mobility) {
            Map<String, NodeGroup> retval = new TreeMap<>();

            String allgroups = configfile.getProperty(GROUPS_KEY);

            if (allgroups != null) {

                for (String groupname : LIST_SEPARATOR.split(allgroups)) {

                    // Parse the list of nodes
                    List<Integer> members = parseNodeList(configfile
                            .getProperty(GROUP_NODES_KEY_PREFIX + groupname));

                    if (members.isEmpty()) {
                        throw new IllegalArgumentException(
                                "No members for group " + groupname);
                    } else {
                        // The new group has no bounding box by default
                        NodeGroup group = new NodeGroup(groupname,
                                members.remove(0), mobility.parameterData.x,
                                mobility.parameterData.y);
                        group.addNodes(members);

                        // Look for a bounding box
                        String boxconf = configfile.getProperty(
                                GROUP_BOUNDING_BOX_KEY_PREFIX + groupname);

                        if (boxconf != null) {
                            group.setBoundingBox(boxconf, mobility.heightMap);
                        }

                        // Put the group in the map
                        if (retval.put(groupname, group) != null) {
                            throw new RuntimeException(
                                    "There is more than one group named "
                                            + groupname);
                        }
                    }
                }
            }

            return retval;
        }
    }

    protected Map<String, NodeGroup> groupMembershipTable = new HashMap<>();
    protected List<StationaryNode> stationaryNodes = new ArrayList<>();
    protected int numSubseg = 4;
    protected double speedScale = 1.5;
    protected double minpause = 0.0;

    protected HeightMap heightMap = null;
    protected String heightMapPath = null;
    protected PositionGeo referencePositionGeo = null;
    protected String groupMembershipPath = null;

    /** Maximum deviation from group center [m]. */
    protected double maxdist = 2.5;

    public HeightMapRNGM(int nodes, double x, double y, double duration,
            double ignore, long randomSeed, double minspeed, double maxspeed,
            double maxpause, double maxdist, double avgMobileNodesPerGroup,
            double groupSizeDeviation, double pGroupChange) {
        super(0, x, y, duration, ignore, randomSeed, minspeed, maxspeed,
                maxpause);

        this.maxdist = maxdist;
        generate();
    }

    public HeightMapRNGM(String[] args) {
        go(args);
    }

    public void go(String args[]) {
        super.go(args);
        generate();
    }

    /**
     * Generate the mobile and stationary nodes for the given parameters
     */
    public void generate() {

        if (minpause > maxpause) {
            System.err.println("Minimum pause " + minpause
                    + " is greater than maximum pause " + maxpause);
            System.exit(-1);
        }

        if (heightMapPath != null) {
            heightMap = new HeightMap(heightMapPath, referencePositionGeo);
        }

        if (groupMembershipPath == null) {
            System.err.println("Group membership file not specified");
            System.exit(-1);
        } else if (!readNodeGroups(groupMembershipPath)) {
            System.exit(-1);
        }

        if (groupMembershipTable.isEmpty()) {
            System.err.println("Group membership table is empty");
            System.exit(-1);
        } else {
            generateForExplicitlyDefinedNodes();
        }
    }

    /**
     * Update the height of the position from the heightMap.
     *
     * @param position
     *            The position to update
     * @return The position with updated height (not a new position)
     */
    private Position updateHeight(Position position) {
        Position retval = position;

        if (heightMap != null) {
            retval.z = heightMap.getHeight(retval);
        }

        return retval;
    }

    /**
     * Create a new position with the correct height
     *
     * @param x
     *            The X-coordinate of the position
     * @param y
     *            The Y-coordinate of the position
     * @return A new position at (x, y) with updated height
     */
    private Position newPosition(double x, double y) {
        return updateHeight(new Position(x, y));
    }

    /**
     * Call {@link Position#rndprox} on the position and update the height on
     * the returned value.
     */
    private Position rndprox(Position position, double maxdist, double dist,
            double dir, Dimension dim) {
        return updateHeight(position.rndprox(maxdist, dist, dir, dim));
    }

    private double randomInRange(double low, double high) {
        double retval = Math.min(low, high) + Math.abs(high - low) / 2;

        if (high > low) {
            retval = low + randomNextDouble(high - low);
        }

        return retval;
    }

    /**
     * Return a random position in the box bounded by ll and ur at the corners
     * that is at least maxdist away from the box's edges
     *
     * @param ll
     *            The lower left corner of the bounding box
     * @param ur
     *            The upper right corner of the bounding box
     * @return a new random position in the box. If the box is smaller than
     *         maxdist in either dimension, return the center of that dimension.
     */
    private Position newPositionInBox(Position ll, Position ur) {

        return newPosition(randomInRange(ll.x + maxdist, ur.x - maxdist),
                randomInRange(ll.y + maxdist, ur.y - maxdist));
    }

    /**
     * Generate motion for a reference node using the random waypoint model
     *
     * @param ll
     *            The lower left corner of the bounding box of the group
     * @param ur
     *            The upper right corner of the bounding box of the group
     * @return The reference node.
     */
    private GroupNode generateForReferenceNode(Position ll, Position ur) {
        GroupNode retval = new GroupNode(null);
        retval.setgroup(retval);

        double t = 0.0;

        // pick position inside the interval
        // [maxdist; x - maxdist], [maxdist; y - maxdist]
        // (to ensure that the group area doesn't overflow the borders)
        Position src = newPositionInBox(ll, ur);

        if (!retval.add(0.0, src)) {
            System.err.println(getInfo().name
                    + ".generate: error while adding reference node "
                    + "movement (1)");
            System.exit(-1);
        }

        while (t < parameterData.duration) {
            Position dst = newPositionInBox(ll, ur);

            double speed = (maxspeed - minspeed) * randomNextDouble()
                    + minspeed;

            t += src.distance(dst) / speed;

            if (!retval.add(t, dst)) {
                System.err.println(getInfo().name
                        + ".generate: error while adding reference node "
                        + "movement (2)");
                System.exit(-1);
            }

            if ((t < parameterData.duration) && (maxpause > 0.0)) {
                double pause = minpause
                        + (maxpause - minpause) * randomNextDouble();

                if (pause > 0.0) {
                    t += pause;

                    if (!retval.add(t, dst)) {
                        System.err.println(getInfo().name
                                + ".generate: error while adding reference "
                                + "node movement (3)");
                        System.exit(-1);
                    }
                }
            }
            src = dst;
        }
        return retval;
    }

    /**
     * Allocate the array of MobileNodes that will be generated in the end.
     * Since the node groups file may contain node IDs that are not sequential
     * from 0 to the number of nodes, this function may create an array that
     * contains nodes for which group motion is not defined. Those nodes should
     * be given a position of the referencePoint
     *
     * @return An array of GroupNodes in which to fill with group nodes.
     */
    private MobileNode[] allocateNodes() {

        Set<Integer> allNodeIds = new HashSet<Integer>();

        for (NodeGroup group : groupMembershipTable.values()) {
            allNodeIds.add(group.getLeader());
            allNodeIds.addAll(group.getNodes());
        }

        for (StationaryNode node : stationaryNodes) {
            allNodeIds.add(node.getId());
        }

        int maxNodeId = Collections.max(allNodeIds);

        MobileNode[] retval = new MobileNode[maxNodeId + 1];

        for (int i = 0; i < retval.length; ++i) {
            if (!allNodeIds.contains(i)) {
                retval[i] = new GroupNode(null);

                retval[i].add(0.0, newPosition(0.0, 0.0));
            }
        }

        return retval;
    }

    /**
     * Determine whether the group member should pause based on the movement of
     * the reference point.
     *
     * @param node
     *            The GroupNode for which to ask the question
     * @param timeindex
     *            The index into GropuNode#changeTimes for which to determine
     *            whether to pause
     * @return true if the node should pause at this time, false otherwise
     */
    private boolean groupMemberShouldPause(GroupNode node, int timeindex) {
        boolean pause = (timeindex == 0);

        MobileNode group = node.group();
        final double[] groupChangeTimes = group.changeTimes();

        if (!pause) {
            final Position pos1 = group
                    .positionAt(groupChangeTimes[timeindex - 1]);
            final Position pos2 = group.positionAt(groupChangeTimes[timeindex]);
            pause = pos1.equals(pos2);
        }

        return pause;
    }

    /**
     * Generate interim movement for a member of a group. Generate numSubseg
     * motions for the node between the motions of the group reference point at
     * the given timeindex.
     *
     * @param node
     *            The node for which to generate movement
     * @param timeindex
     *            The index into the node's group's table of changeTimes
     * @param msrc
     *            The starting position of the interim movement
     * @return The final position of the node.
     */
    private Position generateInterimForGroupMember(GroupNode node,
            int timeindex, Position msrc) {
        Position mdst = null;
        Position prevLoc = msrc;

        MobileNode group = node.group();
        final double[] groupChangeTimes = group.changeTimes();

        Position grpSegStart = group
                .positionAt(groupChangeTimes[timeindex - 1]);

        Position grpSegEnd = group.positionAt(groupChangeTimes[timeindex]);

        double segTimeDelta = (groupChangeTimes[timeindex]
                - groupChangeTimes[timeindex - 1]) / numSubseg;

        for (int segm = 1; segm <= numSubseg; ++segm) {

            double interimX = grpSegStart.x
                    + segm * (grpSegEnd.x - grpSegStart.x) / numSubseg;
            double interimY = grpSegStart.y
                    + segm * (grpSegEnd.y - grpSegStart.y) / numSubseg;

            Position grpSegInterim = newPosition(interimX, interimY);

            double interimTime = groupChangeTimes[timeindex - 1]
                    + segm * segTimeDelta;

            double speed = 0;

            do {
                mdst = rndprox(grpSegInterim, maxdist, randomNextDouble(),
                        randomNextDouble(), parameterData.calculationDim);
                speed = prevLoc.distance(mdst) / segTimeDelta;
            } while (speed > maxspeed * speedScale);

            prevLoc = mdst;

            if (!node.add(interimTime, mdst)) {
                System.err.println(getInfo().name
                        + ".generate: error while adding node movement for "
                        + "intermediate dest");
                System.exit(-1);
            }
        }

        return mdst;
    }

    /**
     * Generate the motion for the GroupNode. The node will have interim motion
     * between the waypoints of the reference point.
     *
     * @param node
     *            The node for which to generate movements.
     */
    private void generateForGroupMember(GroupNode node) {
        double mt = 0.0;

        MobileNode group = node.group();

        Position msrc = rndprox(group.positionAt(mt), maxdist,
                randomNextDouble(), randomNextDouble(),
                parameterData.calculationDim);

        // System.out.println("src: " + msrc.toString());

        if (!node.add(0.0, msrc)) {
            System.err.println(getInfo().name
                    + ".generate: error while adding node movement (1)");
            System.exit(-1);
        }

        while (mt < parameterData.duration) {
            Position mdst = newPosition(0.0, 0.0);
            final double[] groupChangeTimes = group.changeTimes();
            int currentGroupChangeTimeIndex = 0;

            // Determine the current time index for which the member
            // node should move
            while ((currentGroupChangeTimeIndex < groupChangeTimes.length)
                    && (groupChangeTimes[currentGroupChangeTimeIndex] <= mt))
                currentGroupChangeTimeIndex++;

            double next = (currentGroupChangeTimeIndex < groupChangeTimes.length)
                    ? groupChangeTimes[currentGroupChangeTimeIndex]
                    : parameterData.duration;

            boolean pause = groupMemberShouldPause(node,
                    currentGroupChangeTimeIndex);

            // don't do any movement within a pause
            if (pause) {
                mdst = msrc;
            } else {
                mdst = generateInterimForGroupMember(node,
                        currentGroupChangeTimeIndex, msrc);
            }

            msrc = mdst;
            mt = next;
        } // end of while mt < parameterData.duration
    }

    /**
     * For each node, add waypoints between the node's waypoints to account for
     * correct changes of height above datum.
     *
     * @param nodes
     *            The nodes whose way points to process
     * @return The array of nodes with intercalated points.
     */
    private MobileNode[] intercalateHeightWayPoints(MobileNode[] nodes) {
        MobileNode[] retval = nodes;

        if (heightMap != null) {
            retval = new MobileNode[nodes.length];

            for (int i = 0; i < nodes.length; ++i) {
                // Create a new node
                MobileNode newnode = new MobileNode();
                retval[i] = newnode;

                // Process each way point
                Waypoint prevWaypoint = null;
                for (Waypoint waypoint : nodes[i].getWaypoints()) {
                    if (prevWaypoint == null
                            || prevWaypoint.pos.equals(waypoint.pos)) {
                        // If there was no movement, copy the point
                        newnode.add(waypoint.time, waypoint.pos);
                    } else {
                        // If there way movement, get the path
                        List<Position> path = heightMap
                                .getPath(prevWaypoint.pos, waypoint.pos);
                        double distance = heightMap.getLength(path);
                        path.remove(0); // The first point on the path is the
                                        // previous way point, so remove it.

                        // Use the average speed
                        double speed = distance
                                / (waypoint.time - prevWaypoint.time);

                        double t = prevWaypoint.time;
                        Position prevPosition = prevWaypoint.pos;

                        for (Position position : path) {
                            t += prevPosition.distance(position) / speed;
                            newnode.add(t, position);
                            prevPosition = position;
                        }
                    }
                    prevWaypoint = waypoint;
                }
            }
        }

        return retval;
    }

    /**
     * Generate for mobile and stationary nodes that are defined
     */
    private void generateForExplicitlyDefinedNodes() {
        preGeneration();

        MobileNode[] nodes = allocateNodes();

        generateStationaryNodes(nodes);
        generateGroupNodes(nodes);

        this.parameterData.nodes = intercalateHeightWayPoints(nodes);

        postGeneration();
    } // end of generateForExplicitlyDefinedGroups method

    /**
     * Generates for the group nodes that are defined. For each group make the
     * first node a leader, and all the others a member of the group they belong
     * to.
     */
    private void generateGroupNodes(MobileNode[] nodes) {
        for (NodeGroup group : groupMembershipTable.values()) {
            int groupId = group.getLeader();
            List<Integer> members = group.getNodes();

            GroupNode ref = generateForReferenceNode(group.getLl(),
                    group.getUr());

            nodes[groupId] = ref;

            for (int memberId : members) {
                GroupNode memberNode = new GroupNode(ref);

                generateForGroupMember(memberNode);

                nodes[memberId] = memberNode;

            } // end of iteration through nodes of a group

        } // end of iteration through group leaders
    }

    /**
     * Generates the stationary nodes that are defined. For each stationary node
     * convert it to a mobile node. Convert the Latitude and Longitude to offset
     * points. The height of the node is taken from the Height Map, and the
     * optional altitude is added on if it defined. Remove any nodes that
     * previously had the same id, and insert the new node into its correct spot
     * in the node list
     */
    private void generateStationaryNodes(MobileNode[] nodes) {

        for (StationaryNode node : this.stationaryNodes) {
            MobileNode newNode = new MobileNode();
            PositionGeo geoPosition = node.getPosition();
            Position position = heightMap
                    .transformFromWgs84ToPosition(geoPosition);

            position.z = this.heightMap.getHeight(position);
            if (node.getAltitude() != null) {
                position.z += node.getAltitude();
            }
            newNode.add(0, position);
            nodes[node.getId()] = newNode;
        }
    }

    /**
     * Clears the groupMembershipTable and adds all the groups from the
     * configuration file
     *
     * @param configfile
     *            the properties file containing group information
     * 
     * @return true if reading was successful
     */
    private boolean readGroupMembership(Properties configfile) {
        boolean retval = true;

        groupMembershipTable.clear();
        try {
            groupMembershipTable
                    .putAll(NodeGroup.parseGroups(configfile, this));

        } catch (Exception e) {
            retval = false;
            System.err.println("Unable to read group membership: " + e);
        }

        return retval;
    }

    /**
     * Reads the config file and adds the mobile and stationary nodes to
     * separate groups
     *
     * @param nodeGroupsPath
     *            File path to the configuration
     * @return true if successful configuration of mobile and stationary nodes
     */
    private boolean readNodeGroups(String nodeGroupsPath) {

        boolean retval = true;

        try {
            Properties nodeGroupsFile = new Properties();

            nodeGroupsFile.load(new FileInputStream(nodeGroupsPath));

            retval &= readGroupMembership(nodeGroupsFile);
            retval &= readStationaryNodes(nodeGroupsFile);
        } catch (java.io.FileNotFoundException fnfe) {
            retval = false;
            System.err.println(
                    "Unable to read group membership: " + fnfe.getMessage());
        } catch (java.io.IOException ioe) {
            retval = false;
            System.err.println(
                    "Unable to read group membership: " + ioe.getMessage());
        }

        return retval;
    }

    /** The prefix of the keys that set static positions for nodes */
    private static final Pattern STATIONARY_NODE_RE = Pattern
            .compile("position_node_(\\d+)");

    /**
     * Clears stationaryNodes and adds all the stationary nodes from the
     * configuration file
     *
     * @param configfile
     *            The parsed configuration file containing the static nodes
     * @return true if reading was successful
     */
    private boolean readStationaryNodes(Properties configfile) {

        boolean retval = true;

        this.stationaryNodes.clear();

        try {

            for (Map.Entry<Object, Object> entry : configfile.entrySet()) {

                Matcher m = STATIONARY_NODE_RE.matcher((String) entry.getKey());

                if (m.matches()) {

                    boolean addNode = true;

                    int nodeid = Integer.parseInt(m.group(1), 10);
                    String[] value = ((String) entry.getValue()).split("\\s+");
                    Double altitude = null;

                    switch (value.length) {
                    case 1:
                        break;
                    case 2:
                        altitude = Double.parseDouble(value[1]);
                        break;
                    default:
                        System.err.println("Could not parse position for node "
                                + nodeid + ": too many parameters");
                        addNode = false;
                    }

                    if (addNode) {
                        this.stationaryNodes
                                .add(StationaryNode.createNode(
                                        nodeid, PositionGeoParser
                                                .parsePositionGeo(value[0]),
                                        altitude));
                    }

                    retval &= addNode;
                }
            }

            this.stationaryNodes.sort(StationaryNode::compareById);

        } catch (Exception e) {
            System.err.println(
                    "Unable to read stationary nodes: " + e.getMessage());
        }

        return retval;
    }

    /**
     * Parse the arguments from the parameters file generated by
     * {@link #write(String)}
     *
     * @param key
     *            The name of the parameter
     *
     * @param value
     *            The value of the parameter
     */
    @Override
    protected boolean parseArg(String key, String value) {
        if (key.equals("maxdist")) {
            maxdist = Double.parseDouble(value);
            return true;
        } else if (key.equals("numSubseg")) {
            numSubseg = Integer.parseInt(value);
            return true;
        } else if (key.equals("speedScale")) {
            speedScale = Double.parseDouble(value);
            return true;
        } else if (key.equals("minpause")) {
            minpause = Double.parseDouble(value);
            return true;
        } else if (key.equals("origin")) {
            referencePositionGeo = PositionGeoParser.parsePositionGeo(value);
            return true;
        } else
            return super.parseArg(key, value);
    }

    /**
     * Write the scenario properties to a file that can be parsed later
     *
     * @param _name
     */
    @Override
    public void write(String _name) throws FileNotFoundException, IOException {
        String[] p = new String[] { "maxdist=" + maxdist,
                "numSubseg=" + numSubseg, "speedScale=" + speedScale,
                "origin=" + PositionGeoParser.toString(referencePositionGeo),
                "minpause=" + minpause };

        super.write(_name, p);
    }

    /**
     * Parse the command-line arguments to the module
     *
     * @param key
     *            The option (flag)
     * @param val
     *            The value of the argument
     */
    @Override
    protected boolean parseArg(char key, String val) {
        switch (key) {
        case 'e':
            numSubseg = Integer.parseInt(val);
            return true;
        case 'g':
            groupMembershipPath = val;
            // Will be overwritten in generate()
            parameterData.nodes = new MobileNode[0];
            return true;
        case 'm':
            speedScale = Double.parseDouble(val);
            return true;
        case 'n':
            System.err.println("Cannot specify the number of nodes and "
                    + "their groups simultaneously");
            return false;
        case 'o':
            referencePositionGeo = PositionGeoParser.parsePositionGeo(val);
            return true;
        case 'P':
            minpause = Double.parseDouble(val);
            return true;
        case 'r':
            maxdist = Double.parseDouble(val);
            return true;
        case 't':
            heightMapPath = val;
            return true;
        default:
            return super.parseArg(key, val);
        }
    }

    /**
     * Print the help message that lists the command-line arguments
     */
    public static void printHelp() {
        System.out.println(getInfo().toDetailString());
        RandomSpeedBase.printHelp();
        System.out.println(getInfo().name + ":");
        System.out.println("\t-e <num. sub-segments>");
        System.out.println("\t-g <group membership file>");
        System.out.println(
                "\t-m <max speed scale for member speed relative to leader speed>");
        System.out.println("\t-o <origin geo. location>");
        System.out.println("\t-P <min. pause time>");
        System.out.println("\t-r <max. distance to group leader>");
        System.out.println("\t-t <terrain model file>");
    }
}
