package hierarchic;

//yarn jar ~/tmp/kmeans.jar hierarchic.Main /worldcitiespop.txt output 10 3 5 6

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Cluster;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class Main {

	// récupère les clusterPoints depuis le fichier cache distribué sous forme d'une table de hachage
	private static Map<String, ArrayList<ClusterPointWritable>> getClusterPointsFromCache(URI distribURI, Configuration conf, int nbClusters) throws IOException {
		FileSystem fs = FileSystem.get(distribURI, conf);
		InputStream is = fs.open(new Path(distribURI));
		Map<String, ArrayList<ClusterPointWritable>> clusterPoints = ClusterPointWritable.readFromFile(is, nbClusters);
		is.close();
		return clusterPoints;
	}

	// écrit les clusterPoints sous forme d'une table de hachage dans le fichier cache distribué
	private static void writeClusterPointsIntoCache(URI distribURI, Configuration conf, Map<String, ArrayList<ClusterPointWritable>> clusterPoints) throws IOException {
		FileSystem fs = FileSystem.get(distribURI, conf);
		FSDataOutputStream os = fs.create(new Path(distribURI));
		ClusterPointWritable.writeIntoFile(clusterPoints, os);
		os.close();
	}

	public static class HierarchicMapper extends Mapper<LongWritable, Text, Text, PointXDWritable> {
		private Map<String, ArrayList<ClusterPointWritable>> clusterPoints;
		private int nbDimensions;
		private int nbIterations;
		private int nbHierarchies;
		private int[] positions;
		private PointXDWritable currentPoint;
		private ClusterPointWritable nearestClusterPoint;
		private double minDistance;
		private double currentDistance;
		private boolean firstIteration = true;
		private Configuration conf;
		private BufferedWriter bw;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			// récupération des paramètres utiles depuis la configuration et des clusterPoints depuis le cache
			conf = context.getConfiguration();
			this.nbDimensions = conf.getInt("megaProject.args.nbDimensions", 2);
			this.nbIterations = conf.getInt("megaProject.args.nbIterations", 1);
			this.nbHierarchies = conf.getInt("megaProject.args.nbHierarchies", 1);
			this.positions = new int[this.nbDimensions];
			for(int i = 0 ; i<this.nbDimensions;++i)
				this.positions[i] = conf.getInt("megaProject.args.position"+ i, 0);
			ClusterPointWritable.setNbDimensions(this.nbDimensions);
			ClusterPointWritable.setNbIterations(this.nbIterations);
			ClusterPointWritable.setRealNbClusters(conf.getInt("megaProject.args.realNbClusters", 5));
			this.clusterPoints = getClusterPointsFromCache(context.getCacheFiles()[0], conf, conf.getInt("megaProject.args.nbClusters", 10));
		}

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {	
			// on saute la première ligne du fichier "worldcitiespop.txt"
			if(!value.toString().contains("Country")) {
				
				// création du fichier bloc à la première itération
				if(this.firstIteration) {
					FileSystem fs = FileSystem.get(conf);
					String nameFile = "file" + key.get();
					FSDataOutputStream os = fs.create(new Path("file" + key.get()));
					context.getCounter("filenames", nameFile);
					this.bw = new BufferedWriter(new OutputStreamWriter(os));
					this.firstIteration = false;
				}

				// récupération du tableau des clusterPoints (à la première itération, la clé par défaut est "0")
				List<ClusterPointWritable> clusterPointsArray = this.clusterPoints.get(this.nbIterations == 1 ? "0" : ClusterPointWritable.fileLineToKey(value.toString()));

				// détermination du clusterPoint le plus proche du point courant
				this.currentPoint = new PointXDWritable(value.toString(), this.nbIterations, this.positions);
				this.minDistance = Double.MAX_VALUE;
				for(ClusterPointWritable clusterPoint : clusterPointsArray) {
					this.currentDistance = this.currentPoint.distance(clusterPoint);
					if(this.currentDistance < this.minDistance) {
						this.minDistance = this.currentDistance;
						this.nearestClusterPoint = clusterPoint;
					}
				}

				// écriture en sortie et dans le fichier du bloc courant
				context.write(new Text(this.nearestClusterPoint.toString()), this.currentPoint);
				if(this.nbIterations == this.nbHierarchies) {
					String str = value.toString().replace(':', ',');
					bw.write(str + "," + this.nearestClusterPoint.getLastIndex() + "\n");
				}
				else
					bw.write(value.toString() + ":" + this.nearestClusterPoint.getLastIndex() + "\n");
				bw.flush();
			}
		}
	}

	public static class HierarchicReducer extends Reducer<Text, PointXDWritable, PointXDWritable, Text> {
		private Map<String, ArrayList<ClusterPointWritable>> clusterPoints;
		private double[] coords;
		private ClusterPointWritable newCenter;
		private int nbPoints;
		private int nbDimensions;
		private int nbIterations;
		private int nbHierarchies;
		private Configuration conf;
		private List<String> filenames = new ArrayList<String>();
		private Path[] paths;

		// concatène les fichiers intermédiaires à la fin d'une boucle de stabilisation
		private void concatFiles() throws IOException {
			FileSystem fs = FileSystem.get(this.conf);
			Path endFilePath;
			
			if(this.nbIterations != this.nbHierarchies) // cas où on a pas encore terminé, on écrit dans le fichier intermédiaire
				endFilePath = new Path(conf.get("megaProject.args.resultFile", "resultEven"));
			else // dernière itération, on écrit dans le fichier de sortie
				endFilePath = new Path(conf.get("megaProject.args.outputFile"));
			fs.create(endFilePath).close();
			this.paths = new Path[this.filenames.size()];
			for(int i = 0; i < this.paths.length; ++i)
				this.paths[i] = new Path(this.filenames.get(i));
			fs.concat(endFilePath, this.paths);
			for(int i = 0; i < this.paths.length; ++i)
				fs.delete(this.paths[i], true); // suppression des fichiers blocs
		}

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			conf = context.getConfiguration();
			
			// récupération des compteurs et récupération des noms des fichiers intermédiaires grâce à leur nom de famille de compteur
			// un fichier intermédiaire correspond à un bloc de données
			Cluster cluster = new Cluster(this.conf);
			Job job = cluster.getJob(context.getJobID());
			job.getCounters().getGroup("filenames").forEach(counter -> this.filenames.add(counter.getName()));

			// tri des fichiers intermédiaires selon leur nom (le nom de chaque bloc correspondant à la position du premier caractère lu dans le fichier d'entrée)
			this.filenames.sort(new Comparator<String>() {
				public int compare(String o1, String o2) {
					return (int) (extractLong(o1) - extractLong(o2));
				}

				private long extractLong(String s) {
					String num = s.replaceAll("\\D", "");
					return num.isEmpty() ? 0 : Long.parseLong(num);
				}
			});

			// récupération des paramètres utiles depuis la configuration et des clusterPoints depuis le cache
			this.nbDimensions = conf.getInt("megaProject.args.nbDimensions", 2);
			this.nbIterations = conf.getInt("megaProject.args.nbIterations", 1);
			this.nbHierarchies = conf.getInt("megaProject.args.nbHierarchies", 1);
			ClusterPointWritable.setRealNbClusters(conf.getInt("megaProject.args.realNbClusters", 5));
			ClusterPointWritable.setNbDimensions(this.nbDimensions);
			ClusterPointWritable.setNbIterations(this.nbIterations);
			this.clusterPoints = getClusterPointsFromCache(context.getCacheFiles()[0], conf, conf.getInt("megaProject.args.nbClusters", 10));
		}

		@Override
		public void reduce(Text key, Iterable<PointXDWritable> values, Context context) throws IOException, InterruptedException {
			
			// calcul d'un nouveau clusterPoint à l'aide de la liste des points à proxmité
			this.coords = new double[this.nbDimensions];
			this.nbPoints = 0;
			for(PointXDWritable point : values) {
				for(int i = 0; i < this.nbDimensions; ++i)
					this.coords[i] += point.coords[i];
				this.nbPoints++;
			}
			for(int i = 0; i < this.nbDimensions; ++i)
				this.coords[i] /= this.nbPoints;
			this.newCenter = new ClusterPointWritable(this.coords, ClusterPointWritable.lineToIndexes(key.toString()));

			// si un clusterPoint a changé alors on incrémente le compteur de clusterPoints
			String strKey = ClusterPointWritable.indexesToKey(this.newCenter.getIndexes());
			int lastIndex = this.newCenter.getLastIndex();
			if(!this.newCenter.equals(this.clusterPoints.get(strKey).get(lastIndex))) {
				this.clusterPoints.get(strKey).set(lastIndex, newCenter);
				context.getCounter("megaProject.counters", "clusterPointsChanged").increment(1);
			}
		}

		@Override
		public void cleanup(Context context) throws IOException, InterruptedException {
			
			// écriture des nouveaux clusterPoints obtenus dans le fichier cache et concaténation des fichiers blocs
			writeClusterPointsIntoCache(context.getCacheFiles()[0], context.getConfiguration(), this.clusterPoints);
			this.concatFiles();
		}
	}

	// crée un nouveau job en lui paramétrant les différents fichiers à utiliser
	private static Job setJobConfiguration(Configuration conf, FileSystem fs, Path distribPath, Path inputPath, Path outputPath) throws IOException {
		Job job = Job.getInstance(conf, "MegaProjectOfTheKillingDeath");

		job.addCacheFile(distribPath.toUri());

		job.setNumReduceTasks(1);
		job.setJarByClass(Main.class);

		job.setMapperClass(HierarchicMapper.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(PointXDWritable.class);

		job.setReducerClass(HierarchicReducer.class);
		job.setOutputKeyClass(PointXDWritable.class);
		job.setOutputValueClass(Text.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(job, inputPath);

		if(fs.exists(outputPath))
			fs.delete(outputPath, true);
		FileOutputFormat.setOutputPath(job, outputPath);

		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		if(args.length < 5) {
			System.out.println("Usage : <command> <inputFile> <outputFile> <nbClusters> <nbHierarchies> <position1> ... <positionN>");
			System.exit(0);
		}

		int nbClusters = Integer.parseInt(args[2]);
		int nbHierarchies = Integer.parseInt(args[3]);
		int nbDimensions = args.length - 4;
		int nbIterations = 1;
		
		// récupération des positions des coordonnées dans une ligne du fichier d'entrée et ajout dans la configuration
		int[] positions = new int[nbDimensions];
		for(int i = 4; i < args.length; ++i) {
			positions[i - 4] = Integer.valueOf(args[i]);
			conf.setInt("megaProject.args.position" + (i - 4), positions[i - 4]);
		}
		
		// mise en place dans la configuration et dans notre classe statique des paramètres utiles
		conf.setInt("megaProject.args.nbClusters", nbClusters);
		conf.setInt("megaProject.args.nbDimensions", nbDimensions);
		conf.setInt("megaProject.args.nbHierarchies", nbHierarchies);
		conf.setInt("megaProject.args.nbIterations", nbIterations);
		conf.set("megaProject.args.outputFile", args[1]);
		ClusterPointWritable.setNbIterations(nbIterations);
		ClusterPointWritable.setNbDimensions(nbDimensions);

		Path inputPath = new Path(args[0]); // fichier d'entrée
		Path outputPath = new Path(args[1]); // fichier de sortie
		Path distribPath = new Path("distributedCache"); // fichier cache distribuée
		Path resultPathEven = new Path("resultEven"); // premier fichier intermédiaire
		Path resultPathUneven = new Path("resultUneven"); // second fichier intermédiaire

		// initialisation des clusterPoints depuis le fichier d'entrée et écriture dans le cache
		Map<String, ArrayList<ClusterPointWritable>> clusterPoints = ClusterPointWritable.initiateClusterPoints(fs.open(inputPath), nbClusters, positions);
		conf.setInt("megaProject.args.realNbClusters", ClusterPointWritable.getRealNbClusters());
		writeClusterPointsIntoCache(distribPath.toUri(), conf, clusterPoints);

		boolean hasFailed = false;
		int clusterPointsChanged = 2;
		Job job;
		long beginTime = new Date().getTime();

		// boucle de construction hiérarchique
		while(nbIterations - 1 < nbHierarchies) {
			conf.setInt("megaProject.args.clusterPointsChanged", 2);
			System.out.println("BEGIN OF HIERARCHIC ITERATION NUMBER " + nbIterations);
			
			// boucle de stabilisation des clusterPoints
			while(clusterPointsChanged > 0) {
				System.out.println("BEGIN OF STABILISATION ITERATION, CLUSTER POINTS CHANGED = " + clusterPointsChanged);
				if(nbIterations == 1) { // à la première itération on lit depuis le fichier d'entrée
					conf.set("megaProject.args.resultFile", "resultEven");
					job = setJobConfiguration(conf, fs, distribPath, inputPath, outputPath);					
				}
				else {
					if(nbIterations % 2 == 1) { // on lit depuis le second fichier intermédiaire et on écrit dans le premier
						conf.set("megaProject.args.resultFile", "resultEven");
						job = setJobConfiguration(conf, fs, distribPath, resultPathUneven, outputPath);
					}
					else { // on lit depuis le premier fichier intermédiaire et on écrit dans le second
						conf.set("megaProject.args.resultFile", "resultUneven");
						job = setJobConfiguration(conf, fs, distribPath, resultPathEven, outputPath);
					}
				}
				hasFailed = job.waitForCompletion(true);
				
				// récupération du nombre de clusterPoints modifiés et affichage de la différence entre les anciens clusterPoints et les nouveaux
				clusterPointsChanged = (int) job.getCounters().findCounter("megaProject.counters", "clusterPointsChanged").getValue();
				conf.setInt("megaProject.args.clusterPointsChanged", clusterPointsChanged);
				
				System.out.println("END OF STABILISATION ITERATION");
			}
			System.out.println("END OF HIERARCHIC ITERATION NUMBER " + nbIterations);
			
			// mise à jour du nombre d'itérations dans notre classe statique et dans la configuration
			nbIterations++;
			ClusterPointWritable.setNbIterations(nbIterations);
			conf.setInt("megaProject.args.nbIterations", nbIterations);

			// récupération des nouveaux clusterPoints depuis le fichier intermédiaire de stabilisation et écriture dans le fichier cache
			if(nbIterations - 1 != nbHierarchies) {
				clusterPoints = ClusterPointWritable.initiateClusterPoints(fs.open(nbIterations % 2 == 1 ? resultPathUneven : resultPathEven), nbClusters, positions);
				conf.setInt("megaProject.args.realNbClusters", ClusterPointWritable.getRealNbClusters());
				writeClusterPointsIntoCache(distribPath.toUri(), conf, clusterPoints);
				clusterPointsChanged = 2;
			}
		}
		
		// suppression des fichiers intermédiaires et affichage du temps d'exécution
		fs.delete(resultPathUneven, true);
		fs.delete(resultPathEven, true);
		System.out.println("Execution time : " + (float) ((new Date().getTime() - beginTime) / 1000) + " seconds.");
		System.exit(hasFailed ? 1 : 0);
	}
}