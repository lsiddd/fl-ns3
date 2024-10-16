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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.gdal.gdal.Band;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;
import org.gdal.osr.CoordinateTransformation;
import org.gdal.osr.SpatialReference;
import org.gdal.osr.osrConstants;

import edu.bonn.cs.iv.bonnmotion.Position;
import edu.bonn.cs.iv.util.maps.PositionGeo;

/**
 * Maps between position and the height on a terrain map. The terrain map is
 * assumed to be raster image whose single raster band contains height values in
 * meters.
 *
 * @author Yitzchak M. Gottlieb <ygottlieb@perspectalabs.com>
 *
 */
public class HeightMap {

    /**
     * The open GDAL Dataset for the terrain map.
     */
    private Dataset dataset = null;
    private Band rasterBand = null;
    private CoordinateTransformation toWgs84 = null;
    private CoordinateTransformation fromWgs84 = null;

    private double[] datasetInvTransform = null;

    private double noDataValue = Double.NaN;

    private Position origin;

    /**
     * The scale of the map: how many meters per unit in the projection of the
     * map
     */
    private double linearScale = 1.0;

    /**
     * Scaling for values read from the data
     */
    private double zScale = 1.0;
    private double zOffset = 0.0;

    // Register all drivers so that GDAL can parse the terrain map
    static {
        gdal.AllRegister();
    }

    /**
     * Apply the inverse transform to convert between the projection's
     * coordinates and the raster position.
     *
     * @param x
     *            The x-coordinate in the map's projection
     * @param y
     *            The y-coordinate in the maps' projection
     * @return the raster point corresponding to the the given coordinate.
     */
    private Position applyInvTransform(double x, double y) {

        double[] retvalX = new double[1];
        double[] retvalY = new double[1];

        gdal.ApplyGeoTransform(datasetInvTransform, x, y, retvalX, retvalY);

        return new Position(retvalX[0], retvalY[0]);
    }

    /**
     * Apply a coordinate transformation on a two dimensional point
     *
     * @param ct
     *            The transformation to apply
     * @param x
     *            The X-coordinate to which to apply the transformation
     * @param y
     *            The Y-coordinate to which to apply the transformation
     * @return The position resulting from the applied transformation
     */
    private Position transformPosition(CoordinateTransformation ct, double x,
            double y) {
        double[] retval = new double[3];

        retval[0] = x;
        retval[1] = y;
        retval[2] = 0.0;

        ct.TransformPoint(retval);

        return new Position(retval[0], retval[1]);
    }

    /**
     * Transform the position from one projection to another
     *
     * @param ct
     *            The Coordinate Transformation between projections
     * @param position
     *            The position to transform
     * @return The position in the new projection
     */
    private Position transformPosition(CoordinateTransformation ct,
            Position position) {
        return transformPosition(ct, position.x, position.y);
    }

    private Position transformPosition(CoordinateTransformation ct,
            PositionGeo position) {
        return transformPosition(ct, position.x(), position.y());
    }

    /**
     * The corner position at the raster location (x, y)
     *
     * @author ygottlieb
     *
     */
    private class Corner {

        PositionGeo geo;
        int x;
        int y;

        public Corner(int x, int y) {
            this.x = x;
            this.y = y;
            this.geo = HeightMap.this.getPosition(x, y);
        }

        /**
         * @return The position in the projection corresponding to this corner
         */
        public Position getPosition() {
            double[] retvalX = new double[1];
            double[] retvalY = new double[1];

            gdal.ApplyGeoTransform(dataset.GetGeoTransform(), x, y, retvalX,
                    retvalY);

            return new Position(retvalX[0], retvalY[0]);
        }

        @Override
        public String toString() {
            return "(" + x + ", " + y + ") = " + geo;
        }
    }

    /**
     * @return the origin of the map: a position at the southwest corner of the
     *         map
     */
    private final Position getOrigin() {

        List<Corner> positions = Arrays.asList(new Corner(0, 0),
                new Corner(dataset.GetRasterXSize() - 1,
                        dataset.GetRasterYSize() - 1),
                new Corner(dataset.GetRasterXSize() - 1, 0),
                new Corner(0, dataset.GetRasterYSize() - 1));

        Comparator<Corner> minLatitude = new Comparator<Corner>() {

            @Override
            public int compare(Corner o1, Corner o2) {
                return Double.compare(o1.geo.lat(), o2.geo.lat());
            }

        };

        Comparator<Corner> minLongitude = new Comparator<Corner>() {

            @Override
            public int compare(Corner o1, Corner o2) {
                return Double.compare(o1.geo.lon(), o2.geo.lon());
            }

        };
        positions.sort(minLatitude);

        positions = Arrays.asList(positions.get(0), positions.get(1));
        positions.sort(minLongitude);

        return positions.get(0).getPosition();
    }

    /**
     * Is the given position valid in the raster (pixel/line) space
     *
     * @param p
     *            The position to check
     * @return true if each of the position's indices is positive and less than
     *         the maximum index.
     */
    private boolean isValidRaster(Position p) {
        return isValidRaster(p.x, p.y);
    }

    private boolean isValidRaster(double x, double y) {
        return 0 <= x && x < dataset.GetRasterXSize() && 0 <= y
                && y < dataset.GetRasterYSize();
    }

    /**
     * @param position
     *            the geographical position in WGS84
     * @return The projected position of the geographical position onto the
     *         height map's projection.
     * @throws IllegalArgumentException
     *             if the projected position is not in the raster
     *
     **/
    private Position getOrigin(PositionGeo position) {
        Position retval = transformPosition(fromWgs84, position);

        Position checkRaster = applyInvTransform(retval.x, retval.y);

        if (!isValidRaster(checkRaster)) {
            throw new IllegalArgumentException("The geographic position "
                    + position + " is outside the height map");
        }

        return new Position(retval.x, retval.y);
    }

    /**
     * Return the offset position given the geographical coordinate.
     *
     * @param position
     *            The georgraphical coordinate to convert to offset position
     * @return The offset position of the geographical coordinate
     */
    public Position transformFromWgs84ToPosition(PositionGeo position) {
        Position projection = transformPosition(fromWgs84, position);
        return transformFromProjectionToPosition(projection);
    }

    /**
     * Return the raster position given the geographical coordinate
     *
     * @param position
     *            The georgraphical coordinate to convert to raster position
     * @return The raster position of the geographical coordinate
     */
    public Position transformFromWgs84ToRaster(PositionGeo position) {
        Position retval = transformPosition(fromWgs84, position);

        Position checkRaster = applyInvTransform(retval.x, retval.y);

        if (!isValidRaster(checkRaster)) {
            throw new IllegalArgumentException("The geographic position "
                    + position + " is outside the height map");
        }

        return new Position(checkRaster.x, checkRaster.y);
    }

    /**
     * Return the offset position given the projection coordinate
     *
     * @param position
     *            The projection coordinate to convert to offset position
     * @return The offset position of the projection coordinate
     */
    private Position transformFromProjectionToPosition(Position position) {
        double scaledX = (position.x * linearScale) - origin.x;
        double scaledY = (position.y * linearScale) - origin.y;
        Position offset = new Position(scaledX, scaledY);
        return offset;
    }

    /**
     * Return the geographic position in WGS84 corresponding to the given raster
     * indices
     *
     * @param x
     *            the x-coordinate in the raster space
     * @param y
     *            the y-coordinate in the raster space
     * @return The WGS84 geographic position corresponding to the gvein pointer.
     */
    private PositionGeo getPosition(int x, int y) {
        double[] geoX = new double[1];
        double[] geoY = new double[1];

        gdal.ApplyGeoTransform(dataset.GetGeoTransform(), x, y, geoX, geoY);

        Position position = transformPosition(toWgs84, geoX[0], geoY[0]);

        return new PositionGeo(position.x, position.y);
    }

    /**
     * Create a height map from the terrain file
     *
     * @param path
     *            The path to the terrain file
     */
    public HeightMap(String path, PositionGeo origin) {

        // Open the file
        dataset = gdal.Open(path);
        rasterBand = dataset.GetRasterBand(1);

        SpatialReference datasetProjection = new SpatialReference(
                dataset.GetProjection());

        if (!datasetProjection.GetLinearUnitsName().equalsIgnoreCase("Metre")) {
            throw new RuntimeException(
                    "Map " + path + " uses projection units not in meters");
        } else if (datasetProjection.GetAxisOrientation(null,
                0) != osrConstants.OAO_East) {
            throw new RuntimeException("Map " + path
                    + " uses projection with first axis not towards EAST.");
        } else if (datasetProjection.GetAxisOrientation(null,
                1) != osrConstants.OAO_North) {
            throw new RuntimeException("Map " + path
                    + " uses projection with second axis not towards NORTH.");
        }

        double[] datasetTransform = dataset.GetGeoTransform();

        datasetInvTransform = gdal.InvGeoTransform(datasetTransform);

        if (datasetInvTransform == null) {
            throw new RuntimeException(
                    "Map " + path + " uses non-invertable projection");
        }

        linearScale = datasetProjection.GetLinearUnits();

        // Transform to WGS84 since PositionGeo.distance() uses that geodesic.
        SpatialReference wgs84 = new SpatialReference();
        wgs84.SetWellKnownGeogCS("WGS84");
        toWgs84 = CoordinateTransformation.CreateCoordinateTransformation(
                new SpatialReference(dataset.GetProjection()), wgs84);

        fromWgs84 = CoordinateTransformation.CreateCoordinateTransformation(
                wgs84, new SpatialReference(dataset.GetProjection()));

        Double[] read = new Double[1];
        rasterBand.GetNoDataValue(read);
        if (read[0] != null) {
            noDataValue = read[0];
        }

        read[0] = null;
        rasterBand.GetOffset(read);
        if (read[0] != null) {
            zOffset = read[0];
        }

        read[0] = null;
        rasterBand.GetScale(read);
        if (read[0] != null) {
            zScale = read[0];
        }

        if (origin != null) {
            this.origin = getOrigin(origin);
        } else {
            this.origin = getOrigin();
        }
    }

    /**
     * Get the height based on the terrain map at the given location.
     *
     * @param position
     *            The position for which to get the height
     * @return @see {@link #getAveragedHeight(double, double)}
     */
    public double getHeight(Position position) {
        return getAveragedHeight(position.x, position.y);
    }

    /**
     * Read the information from the raster at the given point
     *
     * @param x
     *            The x-coordinate in the raster (pixel/line) space
     * @param y
     *            The y-coordinate in the raster (pixel/line) space
     * @return The value of the raster map at that point. If the raster has no
     *         value return Double.NEGATIVE_INFINITY, if there was an error
     *         return NaN, if there was a warning return
     *         Double.POSITIVE_INFINITY
     */
    private double readRaster(double x, double y) {
        double retval = Double.NaN;

        double[] read = new double[1];

        int error = rasterBand.ReadRaster((int) x, (int) y, 1, 1, read);

        if (error == gdalconstConstants.CE_None) {
            if (read[0] != noDataValue) {
                retval = zScale * read[0] + zOffset;
            } else {
                retval = Double.NEGATIVE_INFINITY;
            }
        } else if (error == gdalconstConstants.CE_Warning) {
            retval = Double.POSITIVE_INFINITY;
        }

        return retval;
    }

    private double readRaster(Position p) {
        return readRaster(p.x, p.y);
    }

    /**
     * Get the height based on the terrain map at the given location.
     *
     * @param x
     *            The distance in meters from the origin along the X axis in the
     *            projection space
     * @param y
     *            The distance in meters from the origin along the Y axis in the
     *            projection space
     * @return The value of the raster map at that point. If the raster has no
     *         value, returns 0
     */
    public double getHeight(double x, double y) {
        double retval = 0.0;

        double scaledX = origin.x + x / linearScale;
        double scaledY = origin.y + y / linearScale;

        double[] rasterX = new double[1];
        double[] rasterY = new double[1];

        gdal.ApplyGeoTransform(datasetInvTransform, scaledX, scaledY, rasterX,
                rasterY);

        retval = readRaster(rasterX[0], rasterY[0]);

        if (Double.isInfinite(retval)) {
            if (retval > 0) {
                System.err
                        .println("HeightMap.getHeight(): warning no values for "
                                + x + ", " + y);

            } else {
                System.err.println("HeightMap.getHeight(): warning using 0 for "
                        + x + ", " + y + ": "
                        + transformPosition(toWgs84, scaledX, scaledY));
            }
        }

        return retval;
    }

    /**
     * Get the height from the terrain map at the given location averaged with
     * the height from the next nearest cell.
     *
     * @param x
     *            The distance in meters from the top "left" point of the
     *            terrain map along the X axis. ((0,0) to (x, 0) in the raster.)
     * @param y
     *            The distance in meters from the top "left" point of the
     *            terrain map along the Y axis. ((0,0) to (0, y) in the raster.)
     * @param epsilon
     *            The minimum difference in distance that is considered
     *            significant
     * @return The value of the raster map at that point. If the raster has
     *         value Integer.MIN_VALUE, returns 0
     */

    final private static double epsilon = 1e-8;

    public double getAveragedHeight(double x, double y, double epsilon) {

        double heightInCell = getHeight(x, y);

        Position rasterPosition = getRasterPoint(x, y);
        Position nextRaster = new Position(rasterPosition);

        // The offsets into the pixel
        double xOffset = Math.IEEEremainder(rasterPosition.x, 1);
        double yOffset = Math.IEEEremainder(rasterPosition.y, 1);

        double absXOffset = Math.abs(xOffset);
        double absYOffset = Math.abs(yOffset);

        if (Math.abs(0.5 - absXOffset) < epsilon
                && Math.abs(0.5 - absYOffset) < epsilon) {
            // Near enough to center, don't average
        } else if (absXOffset < absYOffset) {
            // Closer in x direction, update the raster to the next
            nextRaster.x -= Math.signum(xOffset);
        } else {
            // Closer in y direction, update the raster to the next
            nextRaster.y -= Math.signum(yOffset);
        }

        // The height in the next cell
        double heightInNextCell = heightInCell;

        // Read the raster if it is valid
        if (isValidRaster(nextRaster)) {
            heightInNextCell = readRaster(nextRaster);
        }

        return (heightInNextCell + heightInCell) / 2.0;
    }

    public double getAveragedHeight(double x, double y) {
        return getAveragedHeight(x, y, epsilon);
    }

    public double getAveragedHeight(Position p, double epsilon) {
        return getAveragedHeight(p, epsilon);
    }

    public double getAveragedHeight(Position p) {
        return getAveragedHeight(p.x, p.y);
    }

    /**
     * Get the point on the raster corresponding to given offset from the origin
     * in the projection.
     *
     * @param x
     *            The x-position as an offset (in meters) from the origin in the
     *            projection space.
     * @param y
     *            The x-position as an offset (in meters) from the origin in the
     *            projection space.
     * @return The indices of the raster cell corresponding to the given point
     */
    public Position getRasterPoint(double x, double y) {
        double scaledX = origin.x + x / linearScale;
        double scaledY = origin.y + y / linearScale;

        double[] rasterX = new double[1];
        double[] rasterY = new double[1];

        gdal.ApplyGeoTransform(datasetInvTransform, scaledX, scaledY, rasterX,
                rasterY);

        return new Position(rasterX[0], rasterY[0]);
    }

    /**
     * Get the point on the raster corresponding to given offset from the origin
     * in the projection.
     *
     * @param p
     *            The position as an offset (in meters) from the origin
     * @return The indices of the raster cell corresponding to the
     */
    private Position getRasterPoint(Position p) {
        return getRasterPoint(p.x, p.y);
    }

    private Position getProjectionOffsetPoint(int x, int y) {
        double[] projectionX = new double[1];
        double[] projectionY = new double[1];

        gdal.ApplyGeoTransform(dataset.GetGeoTransform(), x, y, projectionX,
                projectionY);

        return new Position(projectionX[0] * linearScale - origin.x,
                projectionY[0] * linearScale - origin.y);
    }

    /**
     * Return a list of diagonal line segments between the two positions
     *
     * @param p1
     *            The start position in the projection space
     * @param p2
     *            The end position in the projection space
     * @return a (possibly empty) list of line segments
     */
    private List<LineSegment> getEdgeSegments(Position p1, Position p2) {
        List<LineSegment> retval = new ArrayList<LineSegment>();
        Position rasterP1 = getRasterPoint(p1);
        Position rasterP2 = getRasterPoint(p2);

        int minX = (int) (Math.min(rasterP1.x, rasterP2.x));
        int maxX = (int) (Math.max(rasterP1.x, rasterP2.x));
        int minY = (int) (Math.min(rasterP1.y, rasterP2.y));
        int maxY = (int) (Math.max(rasterP1.y, rasterP2.y));

        for (int x = minX; x <= maxX; ++x) {
            for (int y = minY; y <= maxY; ++y) {
                Position tl = getProjectionOffsetPoint(x, y);
                Position tr = getProjectionOffsetPoint(x + 1, y);
                Position bl = getProjectionOffsetPoint(x, y + 1);
                Position br = getProjectionOffsetPoint(x + 1, y + 1);

                retval.add(new LineSegment(tl, br));
                retval.add(new LineSegment(tr, bl));
            }
        }

        return retval;
    }

    /**
     * Add to edgePoints all the points at which the path between p1 and p2
     * intersects each member of edgeSegments sorted by distance from p1
     *
     * @param edgePoints
     *            the output list of positions
     * @param p1
     *            The start of the segment
     * @param p2
     *            The end of the segment
     * @param edgeSegments
     *            The segments against which to test
     */
    private void addEdgePoints(List<Position> edgePoints, final Position p1,
            final Position p2, List<LineSegment> edgeSegments) {
        LineSegment path = new LineSegment(p1, p2);

        for (LineSegment edgeSegment : edgeSegments) {
            try {
                Position edgePoint = path.GetIntersection(edgeSegment);
                edgePoint.z = Double.NaN;
                edgePoints.add(edgePoint);
            } catch (LineSegment.NoIntersectionException e) {
            } catch (LineSegment.ParallelLinesException e) {
            }
        }

        // Sort edge points
        Comparator<Position> edgePointSorter = new Comparator<Position>() {

            @Override
            public int compare(Position arg0, Position arg1) {
                int retval = 0;
                if (p1.x < p2.x) {
                    retval = Double.compare(arg0.x, arg1.x);
                } else {
                    retval = Double.compare(arg1.x, arg0.x);
                }

                if (retval == 0) {
                    if (p1.y < p2.y) {
                        retval = Double.compare(arg0.y, arg1.y);
                    } else {
                        retval = Double.compare(arg1.y, arg0.y);
                    }
                }

                return retval;
            }
        };

        edgePoints.sort(edgePointSorter);
    }

    /**
     * Get a list of positions on the path between p1 and p2 such that:
     * <ul>
     * <li>p1 and p2 are at the ends of the list</li>
     * <li>positions other than p1 and p2 are either at the edges of a raster
     * pixel (edge point) or at the midpoint between two positions at the edges
     * of a raster pixel (center point).</li>
     * <li>the height of each center point is the value in that raster
     * pixel</li>
     * <li>the height of each edge point is an average of the two adjoining
     * center points</li> </ual>
     *
     * @param p1
     *            The position at the start of the path
     * @param p2
     *            The position at the end of the path
     * @return
     */
    public List<Position> getPath(Position p1, Position p2) {

        List<Position> retval = new ArrayList<Position>();

        // Add all the points along the path at which the height calculation
        // should change.
        addEdgePoints(retval, p1, p2, getEdgeSegments(p1, p2));

        List<Position> centerPoints = new ArrayList<Position>();

        for (int i = 0; i < retval.size() - 1; ++i) {

            Position midPoint = LineSegment.midPoint(retval.get(i),
                    retval.get(i + 1));
            midPoint.z = getHeight(midPoint);
            centerPoints.add(midPoint);
        }

        for (int i = centerPoints.size(); i > 0; --i) {
            retval.add(i, centerPoints.get(i - 1));
        }

        retval.add(0, p1);
        retval.add(p2);

        for (Position position : retval) {
            if (Double.isNaN(position.z)) {
                position.z = getAveragedHeight(position);
            }
        }

        return retval;
    }

    /**
     * Compute the distance between p1 and p2 on the map including the
     * intermediate heights.
     *
     * @param p1
     *            The start position
     * @param p2
     *            The end position of the path
     * @return The length of the path (as computed by
     *         {@link #getPath(Position, Position)}
     */
    public double getDistance(Position p1, Position p2) {
        return getLength(getPath(p1, p2));
    }

    /**
     * Compute the sum of the Euclidian distances between adjacent positions in
     * the list.
     *
     * @param path
     *            The list of positions over which to compute the length.
     * @return Return the computed length of the path
     */
    public double getLength(List<Position> path) {
        double retval = 0.0;
        Position previous = null;

        for (Position position : path) {
            if (previous != null) {
                retval += previous.distance(position);
            }
            previous = position;
        }

        return retval;
    }
}
