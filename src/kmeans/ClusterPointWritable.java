package kmeans;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.IntWritable;

public class ClusterPointWritable extends PointXDWritable {
	private static final double DELTA = 1;
	protected int index;
	
	public ClusterPointWritable() {
		super();
	}
	
	public ClusterPointWritable(double[] coords, int index) {
		super(coords);
		this.index = index;
	}
	
	public ClusterPointWritable(String line) {
		String[] strCoords = line.split("/");
		super.coords = new double[strCoords.length - 1];
		for(int i = 0; i < strCoords.length - 1; ++i)
			super.coords[i] = Double.parseDouble(strCoords[i]);
		this.index = Integer.parseInt(strCoords[strCoords.length - 1]);
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		this.index = in.readByte();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		out.writeByte(this.index);
	}
	
	@Override
	public String toString() {
		StringBuffer str = new StringBuffer();
		for(double coord : super.coords) {
			str.append(coord);
			str.append("/");
		}
		str.append(this.index);
		return str.toString();
	}
	
	@Override
	public boolean equals(Object o) {
		ClusterPointWritable pt  = (ClusterPointWritable) o;
		for(int i = 0; i < this.coords.length; ++i)
			if(this.coords[i] < pt.coords[i] - DELTA || this.coords[i] > pt.coords[i] + DELTA)
				return false;
		return true;
	}
	
	public int compareTo(ClusterPointWritable pt) {
		IntWritable iw1 = new IntWritable(this.index);
		IntWritable iw2 = new IntWritable(pt.index);
		return iw1.compareTo(iw2);
	}
	
	// lit les nbClusters premiers points d'un fichier et les retourne sous la forme d'une liste de ClusterPointWritable
	public static ClusterPointWritable[] initiateClusterPoints(FSDataInputStream inputStream, int nbClusters, int nbDimensions, int[] positions) throws IOException {
		ClusterPointWritable[] clusterPoints = new ClusterPointWritable[nbClusters];
		InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		String line;
		String[] splitLine;
		double[] coords;
		int i = 0;
		
		bufferedReader.readLine(); // pour sauter la premier ligne
		while(i < nbClusters) {
			coords = new double[nbDimensions];
			line = bufferedReader.readLine();
			splitLine = line.split(",");
			for(int j = 0; j < nbDimensions; ++j)
				coords[j] = Double.parseDouble(splitLine[positions[j]]);
			
			if(!coordsAlreadyIn(clusterPoints, coords))
				clusterPoints[i] = new ClusterPointWritable(coords, i++);
		}
		return clusterPoints;
	}
	
	private static boolean coordsAlreadyIn(ClusterPointWritable[] clusterPoints, double[] coords) {
		boolean equal = true;
		for(ClusterPointWritable point : clusterPoints) {
			if(point == null)
				return false;
			equal = true;
			for(int i = 0; i < point.coords.length; ++i)
				if(point.coords[i] != coords[i]) {
					equal = false;
					break;
				}
			if(equal)
				return true;
		}
		return false;
	}
	
	public static void writeIntoFile(ClusterPointWritable[] clusterPoints, OutputStream os) throws IOException {
		OutputStreamWriter osw = new OutputStreamWriter(os);
		BufferedWriter bw = new BufferedWriter(osw);
		StringBuffer strBuffer = new StringBuffer();
		
		for(ClusterPointWritable clusterPoint : clusterPoints) {
			strBuffer.append(clusterPoint.toString());
			strBuffer.append(",");
		}
		
		bw.write(strBuffer.toString());
		bw.close();
		osw.close();
	}
	
	public static ClusterPointWritable[] readFromFile(InputStream is, int nbClusters) throws IOException {
		ClusterPointWritable clusterPoints[] = new ClusterPointWritable[nbClusters];
		InputStreamReader inputStreamReader = new InputStreamReader(is);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		
		String[] points = bufferedReader.readLine().split(",");
		for(int i = 0; i < nbClusters; ++i)
			clusterPoints[i] = new ClusterPointWritable(points[i]);
		
		inputStreamReader.close();
		bufferedReader.close();
		return clusterPoints;
	}
	
	public static void displayClusterPoints(ClusterPointWritable[] clusterPoints) {
		for(ClusterPointWritable point : clusterPoints)
			System.out.println(point);
		System.out.println();
	}
}