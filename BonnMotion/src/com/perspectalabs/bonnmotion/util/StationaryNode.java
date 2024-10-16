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
package com.perspectalabs.bonnmotion.util;

import edu.bonn.cs.iv.util.maps.PositionGeo;

/**
 * Stores information that the user defined within the configuration file.
 * 
 * @author Matthew Witkowski <mwitkowski@perspectalabs.com>
 */

public final class StationaryNode {

    /**
     * The ID of the node
     */
    private final int id;

    /**
     * The geographic position of the node
     */
    private PositionGeo position;

    /**
     * The altitude of the node
     */
    private final Double altitude;

    /**
     * Create a node with ID 0 at the origin
     */
    private StationaryNode() {
        this(0, new PositionGeo(0, 0), 0.0);
    }

    /**
     * Create a StationaryNode
     * 
     * @param id
     *            The ID of the node
     * @param position
     *            The geographic position of the node
     * @param altitude
     *            The altitude of the node
     */
    private StationaryNode(int id, PositionGeo position, Double altitude) {
        this.id = id;
        this.position = position;
        this.altitude = altitude;
    }

    /**
     * Create a StationaryNode,
     * 
     * @see {@link #StationaryNode(int, PositionGeo, Double)}
     * @param id
     *            The ID of the node
     * @param position
     *            The geographic position of the node
     * @param altitude
     *            The altitude of the node
     * @return A new Stationary node
     */
    public static StationaryNode createNode(int id, PositionGeo position,
            Double altitude) {
        return new StationaryNode(id, position, altitude);
    }

    /**
     * @return the ID of the node
     */
    public int getId() {
        return this.id;
    }

    /**
     * @return the geographical position of the node
     */
    public PositionGeo getPosition() {
        return this.position;
    }

    /**
     * @return the altitude of the node
     */
    public Double getAltitude() {
        return this.altitude;
    }

    /**
     * Compare nodes by their ID
     * 
     * @param a
     *            The first node to compare
     * @param b
     *            The second node to compare
     * @return The result of {@link Integer.compare} on the node IDs
     */
    public static int compareById(StationaryNode a, StationaryNode b) {
        return Integer.compare(a.getId(), b.getId());
    }
}
