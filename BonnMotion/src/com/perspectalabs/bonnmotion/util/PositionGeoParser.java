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

import java.math.BigInteger;
import java.text.MessageFormat;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.bonn.cs.iv.util.maps.PositionGeo;

/**
 * Class to parse ISO6709 Annex H strings to PositionGeo objects
 *
 * @author Yitzchak M. Gottlieb <ygottlieb@perspectalabs.com>
 *
 */
public class PositionGeoParser {

    private static Pattern ISO6709_H_LATITUDE = Pattern
            .compile("(?<sign>[+-])" + "(?<deg>\\d{2})" + "((?<min>\\d{2})"
                    + "(?<sec>\\d{2})?)?" + "(\\.(?<fraction>\\d+))?");

    private static Pattern ISO6709_H_LONGITUDE = Pattern
            .compile("(?<sign>[+-])" + "(?<deg>\\d{3})" + "((?<min>\\d{2})"
                    + "(?<sec>\\d{2})?)?" + "(\\.(?<fraction>\\d+))?");

    private static Pattern ISO6709_APPENDIX_H = Pattern
            .compile("(?<latitude>[+-]\\d+(\\.\\d+)?)"
                    + "(?<longitude>[+-]\\d+(\\.\\d+)?)");

    /**
     * Return the named group of the matcher parsed as an Integer
     *
     * @param m
     *            the matcher
     * @param groupname
     *            The name of the group to return from the matcher
     * @return the parsed integer, may be null
     */
    private static BigInteger getPositionPart(Matcher m, String groupname) {
        BigInteger retval = null;

        String group = m.group(groupname);

        if (group != null) {
            retval = new BigInteger(group, 10);
        }

        return retval;
    }

    /**
     * @see #getPositionPart(Matcher, String)
     * @param m
     * @param groupname
     * @param max
     * @return the numeric value of the named group. Double.NaN if the value is
     *         greater than max.
     */
    private static Double getPositionPart(Matcher m, String groupname,
            int max) {
        Double retval;

        BigInteger part = getPositionPart(m, groupname);

        if (part == null) {
            retval = null;
        } else if (part.intValue() <= max) {
            retval = part.doubleValue();
        } else {
            retval = Double.NaN;
        }

        return retval;
    }

    /**
     * Parse the part (latitude or longitude) of the position
     *
     * @param pattern
     *            The pattern with which to parse the part
     * @param part
     *            The part to parse
     * @return The value of the part in the interval [-maxdegree, maxdegree].
     *         Return NaN if cannot be parsed correctly
     */
    private static double parsePositionPart(Pattern pattern, String part,
            int maxdegree) {
        double retval = Double.NaN;

        Matcher m = pattern.matcher(part);

        if (m.matches()) {
            double scale = 1.0;
            retval = getPositionPart(m, "deg", maxdegree);

            Double min = getPositionPart(m, "min", 60);
            if (min != null) {
                retval *= 60.0;
                scale *= 60.0;
                retval += min;
            }

            Double sec = getPositionPart(m, "sec", 60);
            if (sec != null) {
                retval *= 60.0;
                scale *= 60.0;
                retval += sec;
            }

            BigInteger fraction = getPositionPart(m, "fraction");
            if (fraction != null) {
                retval += Double.parseDouble("0." + fraction);
            }

            boolean isnegative = m.group("sign").equals("-");
            retval = retval / scale * (isnegative ? -1.0 : 1.0);
        }

        return retval;
    }

    /**
     * Parse a ISO6709 Annex H position string into a PositionGeo instance
     * 
     * @param position
     *            The string to parse
     * @return A PositionGeo corresponding to the given string
     * @throws IllegalArgumentException
     *             if the string cannot be parsed as an ISO6709 Annex H string.
     */
    public static PositionGeo parsePositionGeo(String position) {
        PositionGeo retval = null;
        Matcher m = ISO6709_APPENDIX_H.matcher(position);

        if (!m.matches()) {
            throw new IllegalArgumentException(
                    "Postion '" + position + "' is no parsable as ISO6709");
        } else {
            double latitude = parsePositionPart(ISO6709_H_LATITUDE,
                    m.group("latitude"), 90);

            double longitude = parsePositionPart(ISO6709_H_LONGITUDE,
                    m.group("longitude"), 180);

            if (Double.isNaN(latitude)) {
                throw new IllegalArgumentException(
                        "Postion '" + m.group("latitude")
                                + "' is not a parsable ISO6709 latitude");
            } else if (Double.isNaN(longitude)) {
                throw new IllegalArgumentException(
                        "Postion '" + m.group("longitude")
                                + "' is not a parsable ISO6709 longitude");
            } else {

                retval = new PositionGeo(longitude, latitude);
            }

            return retval;
        }
    }

    /**
     * Format the of the position to a single string
     * 
     * @param positive
     *            Is the value positive?
     * @param degrees
     *            The number of degrees
     * @param minutes
     *            The number of minutes
     * @param seconds
     *            The number of seconds
     * @param degreeFormat
     *            The format to use for degrees (2 or 3 digits
     * @return A string formatting all the components
     */
    private static String toString(boolean positive, int degrees, int minutes,
            double seconds, String degreeFormat) {

        StringBuilder format = new StringBuilder();
        format.append(positive ? "+" : "-");
        format.append("{0,number,");
        format.append(degreeFormat);
        format.append("}");
        format.append("{1,number,00}");
        format.append("{2,number,00.00000000}");

        return MessageFormat.format(format.toString(), degrees, minutes,
                seconds);
    }

    /**
     * Format the part of the position (latitude or longitude) as a ISO6709
     * string
     * 
     * @param position
     *            A double representing a degree and fractions of a degree
     * @return The part of the position as an ISO6709 string
     */
    private static String toString(double position, String degreeFormat) {
        double sign = Math.signum(position);
        double absposition = Math.abs(position);

        int degrees = Double.valueOf(Math.floor(absposition)).intValue();

        // Get the fraction and shift by 60
        absposition = 60 * (absposition % 1);
        int minutes = Double.valueOf(Math.floor(absposition)).intValue();

        double seconds = 60 * (absposition % 1);

        return toString(sign > 0, degrees, minutes, seconds, degreeFormat);
    }

    /**
     * Format a position as an ISO6709 string
     * 
     * @param position
     *            the position
     * @return the position formatted as an ISO6709 string
     */
    public static String toString(PositionGeo position) {
        StringBuilder retval = new StringBuilder();

        retval.append(toString(position.lat(), "00"));
        retval.append(toString(position.lon(), "000"));

        return retval.toString();
    }
}
