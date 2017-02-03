package kmeans;

// yarn jar ~/tmp/kmeans.jar kmeans.Main /worldcitiespop.txt output 10 5 6

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.Date;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class Main {
	
	private static ClusterPointWritable[] getClusterPointsFromCache(URI distribURI, Configuration conf, int nbClusters) throws IOException {
		ClusterPointWritable[] clusterPoints;
		FileSystem fs = FileSystem.get(distribURI, conf);
		InputStream is = fs.open(new Path(distribURI));
		clusterPoints = ClusterPointWritable.readFromFile(is, nbClusters);
		is.close();
		return clusterPoints;
	}
	
	private static void writeClusterPointsIntoCache(URI distribURI, Configuration conf, ClusterPointWritable[] clusterPoints) throws IOException {
		FileSystem fs = FileSystem.get(distribURI, conf);
		FSDataOutputStream os = fs.create(new Path(distribURI));
		ClusterPointWritable.writeIntoFile(clusterPoints, os);
		os.close();
	}
	
	public static class KmeansWCPMapper extends Mapper<LongWritable, Text, ByteWritable, PointXDWritable> {
		private ClusterPointWritable[] clusterPoints;
		private int nbDimensions;
		private int[] positions;
		private PointXDWritable currentPoint;
		private ClusterPointWritable nearestClusterPoint;
		private double minDistance;
		private double currentDistance;
		private String line;
		
		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			
			this.nbDimensions = conf.getInt("megaProject.args.nbDimensions", 2);
			this.positions = new int[this.nbDimensions];
			for(int i = 0; i < this.nbDimensions; ++i)
				this.positions[i] = conf.getInt("megaProject.args.position" + i, 0);
			PointXDWritable.setNbDimensions(this.nbDimensions);
			this.clusterPoints = getClusterPointsFromCache(context.getCacheFiles()[0], conf, conf.getInt("megaProject.args.nbClusters", 10));
		}
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			this.line = value.toString();
			
			if(!this.line.contains("Country")) {
				this.currentPoint = new PointXDWritable(this.line, this.nbDimensions, this.positions);
				this.minDistance = Double.MAX_VALUE;
				
				for(ClusterPointWritable clusterPoint : this.clusterPoints) {
					this.currentDistance = this.currentPoint.distance(clusterPoint);
					if(this.currentDistance < this.minDistance) {
						this.minDistance = this.currentDistance;
						this.nearestClusterPoint = clusterPoint;
					}
				}
					
				context.write(new ByteWritable((byte) this.nearestClusterPoint.index), this.currentPoint);
			}
		}
	}
	
	public static class KmeansMapper extends Mapper<LongWritable, Text, ByteWritable, PointXDWritable> {
		private ClusterPointWritable[] clusterPoints;
		private int nbDimensions;
		private int[] positions;
		private PointXDWritable currentPoint;
		private ClusterPointWritable nearestClusterPoint;
		private double minDistance;
		private double currentDistance;
		private String line;
		
		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			
			this.nbDimensions = conf.getInt("megaProject.args.nbDimensions", 2);
			this.positions = new int[this.nbDimensions];
			for(int i = 0; i < this.nbDimensions; ++i)
				this.positions[i] = conf.getInt("megaProject.args.position" + i, 0);
			PointXDWritable.setNbDimensions(this.nbDimensions);
			this.clusterPoints = getClusterPointsFromCache(context.getCacheFiles()[0], conf, conf.getInt("megaProject.args.nbClusters", 10));
		}
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			this.line = value.toString();
			
			this.currentPoint = new PointXDWritable(this.line, this.nbDimensions, this.positions);
			this.minDistance = Double.MAX_VALUE;
			
			for(ClusterPointWritable clusterPoint : this.clusterPoints) {
				this.currentDistance = this.currentPoint.distance(clusterPoint);
				if(this.currentDistance < this.minDistance) {
					this.minDistance = this.currentDistance;
					this.nearestClusterPoint = clusterPoint;
				}
			}
				
			context.write(new ByteWritable((byte) this.nearestClusterPoint.index), this.currentPoint);
		}
	}
	
	public static class KmeansCombiner extends Reducer<ByteWritable, PointXDWritable, ByteWritable, PointXDWritable> {
		private double[] coords;
		private PointXDWritable combinerPoint;
		private int nbDimensions;
		private int nbPoints;
		
		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			this.nbDimensions = context.getConfiguration().getInt("megaProject.args.nbDimensions", 2);
			PointXDWritable.setNbDimensions(this.nbDimensions);
		}
		
		@Override
		public void reduce(ByteWritable key, Iterable<PointXDWritable> values, Context context) throws IOException, InterruptedException {
			this.coords = new double[this.nbDimensions];
			this.nbPoints = 0;
			for(PointXDWritable point : values) {
				for(int i = 0; i < this.nbDimensions; ++i)
					this.coords[i] += point.coords[i];
				this.nbPoints += point.getPointsCounter();
			}
			this.combinerPoint = new PointXDWritable(this.coords, this.nbPoints);
			context.write(key, this.combinerPoint);
		}
	}
	
	public static class KmeansReducer extends Reducer<ByteWritable, PointXDWritable, PointXDWritable, Text> {
		private ClusterPointWritable[] clusterPoints;
		private double[] coords;
		private ClusterPointWritable newCenter;
		private int nbDimensions;
		private int nbPoints;
		
		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			this.nbDimensions = conf.getInt("megaProject.args.nbDimensions", 2);
			PointXDWritable.setNbDimensions(this.nbDimensions);
			this.clusterPoints = getClusterPointsFromCache(context.getCacheFiles()[0], conf, conf.getInt("megaProject.args.nbClusters", 10));
		}
		
		@Override
		public void reduce(ByteWritable key, Iterable<PointXDWritable> values, Context context) throws IOException, InterruptedException {
			this.coords = new double[this.nbDimensions];
			this.nbPoints = 0;
			for(PointXDWritable point : values) {
				for(int i = 0; i < this.nbDimensions; ++i)
					this.coords[i] += point.coords[i];
				this.nbPoints += point.getPointsCounter();
			}
			for(int i = 0; i < this.nbDimensions; ++i)
				this.coords[i] /= this.nbPoints;
			
			this.newCenter = new ClusterPointWritable(this.coords, new ClusterPointWritable(key.toString()).index);
			if(!this.newCenter.equals(this.clusterPoints[this.newCenter.index])) {
				this.clusterPoints[this.newCenter.index] = this.newCenter;
				context.getCounter("megaProject.counters", "clusterPointsChanged").increment(1);
			}
		}
		
		@Override
		public void cleanup(Context context) throws IOException, InterruptedException {
			writeClusterPointsIntoCache(context.getCacheFiles()[0], context.getConfiguration(), this.clusterPoints);
		}
	}
	
	private static Job setJobConfiguration(Configuration conf, FileSystem fs, Path distribPath, Path inputPath, Path outputPath) throws IOException {
		Job job = Job.getInstance(conf, "MegaProjectOfTheKillingDeath");
		
		job.addCacheFile(distribPath.toUri());

		job.setNumReduceTasks(1);
		job.setJarByClass(Main.class);
		
		job.setMapperClass(inputPath.getName().equals("worldcitiespop.txt") ? KmeansWCPMapper.class : KmeansMapper.class);
		job.setMapOutputKeyClass(ByteWritable.class);
		job.setMapOutputValueClass(PointXDWritable.class);

		job.setReducerClass(KmeansReducer.class);
		job.setOutputKeyClass(PointXDWritable.class);
		job.setOutputValueClass(Text.class);
		
		job.setCombinerClass(KmeansCombiner.class);
		
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		
		FileInputFormat.addInputPath(job, inputPath);
		
		if(fs.exists(outputPath))
			fs.delete(outputPath,true);
		FileOutputFormat.setOutputPath(job, outputPath);
		
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		
		if(args.length < 5) {
			System.out.println("Usage : <command> <inputFile> <outputFile> <nbClusters> <position1> ... <positionN>");
			System.exit(0);
		}
		
		int nbClusters = Integer.parseInt(args[2]);
		int nbDimensions = args.length - 3;
		int[] positions = new int[nbDimensions];
		for(int i = 3; i < args.length; ++i) {
			positions[i - 3] = Integer.valueOf(args[i]);
			conf.setInt("megaProject.args.position" + (i - 3), positions[i - 3]);
		}
		conf.setInt("megaProject.args.nbClusters", nbClusters);
		conf.setInt("megaProject.args.nbDimensions", nbDimensions);
		
		Path inputPath = new Path(args[0]);
		Path outputPath = new Path(args[1]);
		Path distribPath = new Path("distributedCache");
		
		ClusterPointWritable[] clusterPoints = ClusterPointWritable.initiateClusterPoints(fs.open(inputPath), nbClusters, nbDimensions, positions);
		ClusterPointWritable.displayClusterPoints(clusterPoints);
		
		FileSystem distribFs = FileSystem.get(distribPath.toUri(), conf);
		FSDataOutputStream os = distribFs.create(distribPath);
		ClusterPointWritable.writeIntoFile(clusterPoints, os);
		os.close();
		
		boolean hasFailed = false;
		long clusterPointsChanged = 1;
		Job job;
		
		long startTime = new Date().getTime();
		while(clusterPointsChanged > 0) {
			job = setJobConfiguration(conf, fs, distribPath, inputPath, outputPath);
			hasFailed = job.waitForCompletion(true);
			clusterPointsChanged = job.getCounters().findCounter("megaProject.counters", "clusterPointsChanged").getValue();
			
			ClusterPointWritable.displayClusterPoints(clusterPoints);
			clusterPoints = getClusterPointsFromCache(distribPath.toUri(), conf, nbClusters);
			ClusterPointWritable.displayClusterPoints(clusterPoints);
			
			System.out.println("Execution time : " + ((float) (new Date().getTime() - startTime) / 1000) + " seconds.");
		}
		
		System.exit(hasFailed ? 1 : 0);
	}
}