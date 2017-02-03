package hierarchic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.fs.FSDataInputStream;

public class ClusterPointWritable extends PointXDWritable {
	private static final double DELTA = 5;
	private static int nbIterations;
	private static int realNbClusters;

	private short[] indexes;

	public ClusterPointWritable() {
		super();
	}

	public ClusterPointWritable(double[] coords, short[] indexes) {
		super(coords);
		this.indexes = indexes;
	}

	public ClusterPointWritable(double[] coords, short index) {
		super(coords);
		this.indexes = new short[]{index};
	}

	public short[] getIndexes() {
		return this.indexes;
	}
	
	public short getLastIndex() {
		return this.indexes[this.indexes.length - 1];
	}
	
	public static int getRealNbClusters() {
		return realNbClusters;
	}

	// constructeur qui récupère une ligne de coordonnées et d'indices pour un clusterPoint écrit dans la configuration
	public ClusterPointWritable(String line) {
		String[] fields = line.split(":");

		super.coords = new double[nbDimensions];
		for(int i = 0; i < nbDimensions; ++i)
			super.coords[i] = Double.parseDouble(fields[i]);

		this.indexes = new short[nbIterations];
		for(int i = nbDimensions; i < fields.length; ++i)
			this.indexes[i - nbDimensions] = Short.parseShort(fields[i]);
	}

	public ClusterPointWritable(double[] coords, String key, short index) {
		super(coords);
		this.indexes = new short[nbIterations];
		String[] split = key.split(":");
		for(int i = 0; i < split.length; i++)
			this.indexes[i] = Short.valueOf(split[i]);
		this.indexes[nbIterations - 1] = index;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		this.indexes = new short[nbIterations];
		for(int i = 0; i < nbIterations; ++i)
			this.indexes[i] = in.readShort();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		for(int i = 0; i < nbIterations; ++i)
			out.writeShort(this.indexes[i]);
	}

	// retourne une chaîne de caractères d'un ClusterPointWritable avec chaque coordonnée et indice séparés par un ":"
	@Override
	public String toString() {
		StringBuffer str = new StringBuffer();
		for(double coord : super.coords) {
			str.append(coord);
			str.append(":");
		}
		for(int i = 0; i < nbIterations; ++i) {
			str.append(this.indexes[i]);
			if(i != nbIterations - 1)
				str.append(":");
		}
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

	protected static void setNbIterations(int nbIt) {
		nbIterations = nbIt;
	}

	protected static void setRealNbClusters(int nbClusters) {
		realNbClusters = nbClusters;
	}
	
	// lit les nbClusters premiers points d'un fichier et les retourne sous la forme d'une liste de ClusterPointWritable
	public static Map<String, ArrayList<ClusterPointWritable>> initiateClusterPoints(FSDataInputStream inputStream, int K, int[] positions) throws IOException {
		Map<String, ArrayList<ClusterPointWritable>> clusterPoints = new TreeMap<String, ArrayList<ClusterPointWritable>>();

		InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
		String line;
		String[] splitCoords;
		double[] coords;
		String key;
		int i = 0;
		int nbClusterPoints = (int) Math.pow(K, nbIterations); // nombre théorique de clusterPoints (K * nbIterations)

		bufferedReader.readLine(); // pour sauter la premier ligne
		while(i < nbClusterPoints) {
			if(nbIterations == 1) { // cas où on lit dans le fichier d'entrée
				line = bufferedReader.readLine();
				
				// si la clé n'existe pas encore, alors on lui associe un tableau vide dans la table de hachage
				// comme il n'y a pas encore de clusterPoints définis, on définit la clé à "0"
				key = "0";
				if(clusterPoints.get(key) == null)
					clusterPoints.put(key, new ArrayList<ClusterPointWritable>());
				
				// récupération des coordonnées dans la ligne
				splitCoords = line.split(",");
				coords = new double[nbDimensions];
				for(int j = 0; j < nbDimensions; ++j)
					coords[j] = Double.parseDouble(splitCoords[positions[j]]);
			}
			else { // cas où on lit dans le fichier intermédiaire (fusion des fichiers blocs)
				line = bufferedReader.readLine();
				
				// si on a atteint la fin du fichier, alors le nombre de clusterPoints pour cette hiérarchie sera inférieur à K
				if(line == null)
					break;
				
				// si la clé n'existe pas encore, alors on lui associe un tableau vide dans la table de hachage
				key = fileLineToKey(line);
				if(clusterPoints.get(key) == null)
					clusterPoints.put(key, new ArrayList<ClusterPointWritable>());

				// récupération des coordonnées dans la ligne
				splitCoords = line.split(":")[0].split(",");
				coords = new double[nbDimensions];
				for(int j = 0; j < nbDimensions; ++j)
					coords[j] = Double.parseDouble(splitCoords[positions[j]]);
			}
			
			// si la liste est incomplète et que le point n'existe pas encore dans la liste, il est ajouté à la table de hachage
			if(!coordsAlreadyIn(clusterPoints.get(key), coords, K)) {
				clusterPoints.get(key).add(new ClusterPointWritable(coords, key, (short) clusterPoints.get(key).size()));
				i++;
			}
		}
		realNbClusters = i;
		return clusterPoints;
	}

	// détermine si un point est déjà dans la liste des clusterPoints ou si la liste des clusterPoints est complète ou non
	private static boolean coordsAlreadyIn(ArrayList<ClusterPointWritable> clusterPoints, double[] coords, int K) {
		if(clusterPoints.size() == K)
			return true;
		
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

	// écrit les clusterPoints sous forme d'une table de hachage dans un fichier
	public static void writeIntoFile(Map<String, ArrayList<ClusterPointWritable>> clusterPoints, OutputStream os) throws IOException {
		OutputStreamWriter osw = new OutputStreamWriter(os);
		BufferedWriter bw = new BufferedWriter(osw);
		StringBuffer strBuffer = new StringBuffer();
		for(ArrayList<ClusterPointWritable> arr : clusterPoints.values()) {
			for(ClusterPointWritable clusterPoint : arr) {
				strBuffer.append(clusterPoint.toString());
				strBuffer.append(",");
			}
		}
		bw.write(strBuffer.toString());
		bw.close();
		osw.close();
	}

	// lit depuis un fichier les clusterPoints sous forme de table de hachage
	public static Map<String, ArrayList<ClusterPointWritable>> readFromFile(InputStream is, int K) throws IOException {
		Map<String, ArrayList<ClusterPointWritable>> clusterPoints = new TreeMap<String, ArrayList<ClusterPointWritable>>();
		InputStreamReader inputStreamReader = new InputStreamReader(is);
		BufferedReader bufferedReader = new BufferedReader(inputStreamReader);

		String line = bufferedReader.readLine();
		String[] points = line.split(","); // les points sont séparés par des virgules
		ClusterPointWritable point;
		String key;
		ArrayList<ClusterPointWritable> pointsArray;
		for(int i = 0; i < realNbClusters; ++i) {
			point = new ClusterPointWritable(points[i]);
			key = indexesToKey(point.indexes);
			if(clusterPoints.get(key) == null) {
				(pointsArray = new ArrayList<ClusterPointWritable>(K)).add(point);
				clusterPoints.put(key, pointsArray);
			}
			else	
				clusterPoints.get(key).add(point);
		}

		inputStreamReader.close();
		bufferedReader.close();
		return clusterPoints;
	}

	// affiche la table de hachage contenant les clusterPoints
	public static void displayClusterPoints(Map<String, ArrayList<ClusterPointWritable>> clusterPoints) {
		for(ArrayList<ClusterPointWritable> arr : clusterPoints.values()) {
			for(ClusterPointWritable pt : arr)
				System.out.println(pt);
			System.out.println();
		}
		System.out.println();
	}

	// extrait depuis une ligne de clusterPoint la liste de ses indices
	public static short[] lineToIndexes(String line) {
		String[] fields = line.split(":");

		short[] indexes = new short[nbIterations];
		for(int i = nbDimensions; i < fields.length; ++i)
			indexes[i - nbDimensions] = Short.parseShort(fields[i]);
		return indexes;
	}

	// convertit une liste d'indices en une chaîne de caractères formant une clé de hachage
	public static String indexesToKey(short[] indexes) {
		StringBuilder str = new StringBuilder();
		if(indexes.length - 1 == 0)
			return "0";
		for(int i = 0; i < indexes.length - 1; ++i) {
			str.append(indexes[i]);
			if(i != indexes.length - 2)
				str.append(":");
		}
		return str.toString();
	}

	// convertit une ligne de fichier intermédiaire en une clé de hachage
	public static String fileLineToKey(String value) {
		String[] split = value.split(":");
		StringBuilder str = new StringBuilder();
		for(int i = 1; i < split.length; ++i) {
			str.append(split[i]);
			if(i != split.length - 1)
				str.append(":");
		}
		return str.toString();
	}
}