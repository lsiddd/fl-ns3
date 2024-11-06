#!/bin/bash

BONNMOTION="/home/lucas/fl-ns3/BonnMotion"
PACKAGE="edu.bonn.cs.iv.bonnmotion"
CLASSPATH="/home/lucas/fl-ns3/BonnMotion/classes::/home/lucas/fl-ns3/BonnMotion/lib/gdal.jar:/home/lucas/fl-ns3/BonnMotion/lib/GeographicLib-1.31.jar:/home/lucas/fl-ns3/BonnMotion/lib/jackson-core-2.2.2.jar:/home/lucas/fl-ns3/BonnMotion/lib/mysql-connector-java-5.1.25-bin.jar:/home/lucas/fl-ns3/BonnMotion/lib/osmpbf-1.3.3.jar:/home/lucas/fl-ns3/BonnMotion/lib/proj4j-0.1.0.jar:/home/lucas/fl-ns3/BonnMotion/lib/protobuf-java-2.5.0.jar"
JAVAPATH="/usr/bin"
DOCPATH="/home/lucas/fl-ns3/BonnMotion/javadoc"
JAVA="$JAVAPATH/java"
JAVAC="$JAVAPATH/javac"
JAVADOC="$JAVAPATH/javadoc"
export LD_LIBRARY_PATH="/usr/gdal2/lib64/:/home/lucas/fl-ns3/BonnMotion/lib"

