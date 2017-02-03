package kmeans;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class PointXDWritable implements Writable {
	private static int nbDimensions;
	protected double[] coords;
	private int pointsCounter;
	
	public PointXDWritable() {
		this.coords = new double[nbDimensions];
		this.pointsCounter = 1;
	}
	
	public PointXDWritable(double[] coords) {
		this.coords = coords;
	}
	
	public PointXDWritable(double[] coords, int count) {
		this.coords = coords;
		this.pointsCounter = count;
	}
	
	public PointXDWritable(String line, int nbDimensions, int[] positions) {
		String[] strCoords = line.split(",");
		this.coords = new double[nbDimensions];
		for(int i = 0; i < nbDimensions; ++i)
			this.coords[i] = Double.parseDouble(strCoords[positions[i]]);
		this.pointsCounter = 1;
	}
	
	public int getPointsCounter() {
		return this.pointsCounter;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		for(int i = 0; i < nbDimensions; ++i)
			this.coords[i] = in.readDouble();
		this.pointsCounter = in.readInt();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		for(double coord : this.coords)
			out.writeDouble(coord);
		out.writeInt(this.pointsCounter);
	}
	
	public double distance(PointXDWritable point) {
		double sum = 0;
		for(int i = 0; i < this.coords.length; ++i)
			sum += Math.pow(this.coords[i] - point.coords[i], 2);
		return Math.sqrt(sum);
	}
	
	protected static void setNbDimensions(int nbDim) {
		nbDimensions = nbDim;
	}
}