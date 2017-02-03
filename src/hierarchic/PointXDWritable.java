package hierarchic;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class PointXDWritable implements Writable {
	protected static int nbDimensions;
	
	protected double[] coords;
	
	public PointXDWritable() {
		this.coords = new double[nbDimensions];
	}
	
	public PointXDWritable(double[] coords) {
		this.coords = coords;
	}
	
	public PointXDWritable(String line, int nbIterations, int[] positions) {
		String[] strCoords = nbIterations == 1 ? line.split(",") : line.split(":")[0].split(",");
		this.coords = new double[nbDimensions];
		for(int i = 0; i < nbDimensions; ++i)
			this.coords[i] = Double.parseDouble(strCoords[positions[i]]);
	}

	public void write(DataOutput out) throws IOException {
		for(double coord : this.coords)
			out.writeDouble(coord);
	}
	
	public void readFields(DataInput in) throws IOException {
		for(int i = 0; i < nbDimensions; ++i)
			this.coords[i] = in.readDouble();
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