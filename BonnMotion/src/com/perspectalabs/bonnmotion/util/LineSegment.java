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

import edu.bonn.cs.iv.bonnmotion.Position;

/**
 * A class that represents a line segment with the ability to compute
 * intersections.
 *
 * @author Yitzchak M. Gottlieb <ygottlieb@perspectalabs.com>
 *
 */
public class LineSegment {

    /**
     * How close distances are to be considered 0.
     */
    public static double EPSILON = 1e-5;

    private Position p1;
    private Position p2;

    private double slope;
    private double yIntercept;

    /**
     * Create a new line segment between the positions.
     *
     * @param p1
     *            First position
     * @param p2
     *            Second Position
     */
    public LineSegment(Position p1, Position p2) {
        this.p1 = p1;
        this.p2 = p2;

        if (Math.abs(p2.x - p1.x) < EPSILON) {
            slope = Double.POSITIVE_INFINITY;
        } else {
            slope = (p2.y - p1.y) / (p2.x - p1.x);
        }

        if (Double.isFinite(slope)) {
            yIntercept = p1.y - slope * p1.x;
        }
    }

    /**
     * Exception thrown when two line segments do not intersect, but are not
     * parallel.
     *
     * @author Yitzchak M. Gottlieb <ygottlieb@perspectalabs.com>
     *
     */
    public static class NoIntersectionException extends Exception {

        private static final long serialVersionUID = 5511931877940653676L;
    }

    /**
     * Exception thrown when two line segments do not intersect and are
     * parallel.
     *
     * @author Yitzchak M. Gottlieb <ygottlieb@perspectalabs.com>
     *
     */
    public static class ParallelLinesException extends Exception {

        private static final long serialVersionUID = 3988498536102069282L;
    }

    /**
     * Compute the ordinate corresponding to the abscissa on the line. The
     * returned value may not be on the segment.
     *
     * @param x
     *            the abscissa
     * @return the ordinate on the line of which this segment is a part
     */
    private double GetY(double x) {
        double retval = Double.NaN;

        if (Double.isFinite(slope)) {
            retval = slope * x + yIntercept;
        }

        return retval;
    }

    /**
     * Does the given position lie in the section of the plane bounded by the
     * segement's endpoints.
     *
     * @param p
     *            The position to check
     * @return true if p is bounded by the coordinates of the segment
     */
    private boolean isBetweenEndpoints(Position p) {
        return ((p1.x <= p.x && p.x <= p2.x) || (p2.x <= p.x && p.x <= p1.x))
                && ((p1.y <= p.y && p.y <= p2.y)
                        || (p2.y <= p.y && p.y <= p1.y));
    }

    /**
     * Is this segment parallel to other?
     *
     * @param other
     *            The segment against which to test
     * @return true if the lines are parallel (to within {@link #EPSILON} in the
     *         slopes), false otherwise
     */
    private boolean isParallel(LineSegment other) {
        boolean retval = false;

        if (Double.isInfinite(slope)) {
            retval = Double.isInfinite(other.slope);
        } else if (Double.isInfinite(other.slope)) {
            retval = false;
        } else {
            retval = Math.abs(slope - other.slope) < EPSILON;
        }

        return retval;
    }

    /**
     * Return the intersection point of the two line segments.
     *
     * @param other
     * @return the point on both segments at which the lines intersect
     * @throws NoIntersectionException
     *             if the segments do not intersect
     * @throws ParallelLinesException
     *             if the segments are parallel
     */
    public Position GetIntersection(LineSegment other)
            throws NoIntersectionException, ParallelLinesException {
        Position retval = new Position();

        if (isParallel(other)) {
            throw new ParallelLinesException();
        } else if (Double.isInfinite(slope)) {
            retval = other.GetIntersection(this);
        } else if (Double.isInfinite(other.slope)) {
            retval.x = other.p1.x;
            retval.y = GetY(retval.x);
        } else {
            retval.x = (this.yIntercept - other.yIntercept)
                    / (other.slope - this.slope);
            retval.y = GetY(retval.x);
        }

        if (!isBetweenEndpoints(retval) || !other.isBetweenEndpoints(retval)) {
            throw new NoIntersectionException();
        }

        return retval;
    }

    /**
     * Do the segments intersect?
     *
     * @param other
     * @return true if the segments intersect, false otherwise
     */
    public boolean DoIntersect(LineSegment other) {
        boolean retval = false;

        try {
            GetIntersection(other);
            retval = true;
        } catch (ParallelLinesException e) {
        } catch (NoIntersectionException e) {
        }

        return retval;
    }

    /**
     * @return the midpoint of the segment
     */
    public Position midPoint() {
        return midPoint(p1, p2);
    }

    /**
     * @param p1
     * @param p2
     * @return the point on equidistant from p1 and p2 on the line between them.
     */
    public static Position midPoint(Position p1, Position p2) {
        return new Position((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    }

    @Override
    public String toString() {
        return new StringBuilder(p1.toString()).append("->").append(p2)
                .toString();
    }

    /*************************************************************************/
    /*
     * Some tests
     */
    private static void testIntersection() {
        Position p1 = new Position(0, 0);
        Position p2 = new Position(1, 1);
        Position p3 = new Position(1, 0);
        Position p4 = new Position(0, 1);

        LineSegment l1 = new LineSegment(p1, p2);
        LineSegment l2 = new LineSegment(p3, p4);

        try {
            System.out.println(l1.GetIntersection(l2).toString());
            System.out.println(l2.GetIntersection(l1).toString());
        } catch (ParallelLinesException e) {
            System.out.println("Lines are parallel");
        } catch (NoIntersectionException e) {
            System.out.println("Line segments do not intersect");
        }
    }

    private static void testVerticalIntersection() {
        Position p1 = new Position(0.5, 0);
        Position p2 = new Position(0.5, 1);
        Position p3 = new Position(0, 0.3);
        Position p4 = new Position(1, 0);

        LineSegment l1 = new LineSegment(p1, p2);
        LineSegment l2 = new LineSegment(p3, p4);

        try {
            System.out.println(l1.GetIntersection(l2).toString());
            System.out.println(l2.GetIntersection(l1).toString());
        } catch (ParallelLinesException e) {
            System.out.println("Lines are parallel");
        } catch (NoIntersectionException e) {
            System.out.println("Line segments do not intersect");
        }
    }

    private static void testNoIntersection() {
        Position p1 = new Position(0, 0);
        Position p2 = new Position(0, 1);
        Position p3 = new Position(0.1, 0);
        Position p4 = new Position(1, 0);

        LineSegment l1 = new LineSegment(p1, p2);
        LineSegment l2 = new LineSegment(p3, p4);

        try {
            System.out.println(l1.GetIntersection(l2).toString());
            System.out.println(l2.GetIntersection(l1).toString());
        } catch (ParallelLinesException e) {
            System.out.println("Lines are parallel");
        } catch (NoIntersectionException e) {
            System.out.println("Line segments do not intersect");
        }
    }

    public static void main(String argv[]) {
        testIntersection();
        testVerticalIntersection();
        testNoIntersection();
    }
}
