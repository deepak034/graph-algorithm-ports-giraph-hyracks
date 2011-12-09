package org.apache.giraph.benchmark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.giraph.graph.BasicVertex;
import org.apache.giraph.graph.BspUtils;
import org.apache.giraph.graph.Edge;
import org.apache.giraph.graph.GiraphJob;
import org.apache.giraph.graph.MutableVertex;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.graph.VertexReader;
import org.apache.giraph.graph.VertexWriter;
import org.apache.giraph.lib.TextVertexInputFormat;
import org.apache.giraph.lib.TextVertexOutputFormat;
import org.apache.giraph.lib.TextVertexInputFormat.TextVertexReader;
import org.apache.giraph.lib.TextVertexOutputFormat.TextVertexWriter;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.collect.Maps;


/*
 * Type Parameters:
 * I - vertex id
 * V - vertex data
 * E - edge data
 * M - message data
 */
public class KMeansVertex extends
Vertex<LongWritable, Text, FloatWritable, MapWritable>
implements Tool {

	private FloatWritable[] dataPoints;

	public void setDatapoints(FloatWritable[] points) {
		this.dataPoints = points;
	}

	/* How many supersteps to run */
	public static String SUPERSTEP_COUNT = "KMeansVertex.superstepCount";

	/* How many clusters to create */
	public static String NUMBER_OF_CLUSTERS = "KMeansVertex.numberOfClusters";
	
	/* The largest vertex id after reading the data set */
	public static String LARGEST_VERTEX_ID = "KMeansVertex.largestVertexId";
	
	/* The largest vertex id after reading the data set */
	public static String REPRESENTATIVE_VERTEX_ID = "KMeansVertex.representativeVertexId";

	public Float computeDistance(FloatWritable[] source, FloatWritable[] target) {
		// Compute Euclidean distance between two sets of data points
		double sum_of_differences = 0.0;
		for (int i = 0; i < source.length; i++) 
		{
			sum_of_differences += Math.pow(source[i].get() - target[i].get(), 2.0);
		}

		return new Float(Math.sqrt(sum_of_differences));

	}

	public static <T> List<T> randomSample(List<T> items, int m){
		Random random = new Random();
		for(int i=0;i<items.size();i++){
			int pos = i + random.nextInt(items.size() - i);
			T tmp = items.get(pos);
			items.set(pos, items.get(i));
			items.set(i, tmp);
		}
		return items.subList(0, m);
	}

	@Override
	public void compute(Iterator<MapWritable> msgIterator) throws IOException {

		// FIXME: The super step numbers
		/*** Representative Vertex Behavior ***/
		if (getVertexValue().toString() == "R") { 
			
			// Select k random edges and use the data points of the 
			// destination vertices to create the initial random centroids
			if (getSuperstep() == 1) {

				// Send available vertex ids to the randomly selected data nodes
				// to be able to use for creating the initial centroids
				List<LongWritable> randomIds = randomSample(new ArrayList<LongWritable>(destEdgeMap.keySet()), getConf().getInt(NUMBER_OF_CLUSTERS, -1));
				long firstPossibleId = getConf().getLong(LARGEST_VERTEX_ID, -1) + 1;
				int count = 0;
				for(LongWritable id : randomIds) {
					MapWritable msg = new MapWritable();
					msg.put(getVertexValue(), new LongWritable(firstPossibleId + count));
					sendMsg(id, msg);
					count++;
				}

			}

			if (getSuperstep() == 3) {
				// wrap together all the centroid coordinates and send them to data vertices
				MapWritable centroid_coordinates = new MapWritable();
				while (msgIterator.hasNext()) {

					MapWritable message = msgIterator.next();
					// If the message is from another centroid,
					// add coordinates to outgoing coordinate list
					// into map key'd by vertexId.
					if (message.containsKey("C")) {
						centroid_coordinates.put((LongWritable)message.get("vertexId"), 
								(ArrayWritable)message.get("C"));  		
					}

				}

				// Create message container for outgoing centroid coordinates
				MapWritable msg_centroid_coordinates = new MapWritable();
				msg_centroid_coordinates.put(getVertexValue(), 
						centroid_coordinates);
				sendMsgToAllEdges(msg_centroid_coordinates);
			}

			// FIXME Disconnect from data vertices
		}

		/*** Data Vertex Behavior ***/
		if (getVertexValue().toString() == "D") { 
			
			if (getSuperstep() == 0) {
				// add an edgeRequest to create edge from representative vertex to each data vertex
				addEdgeRequest(new LongWritable(getConf().getLong(REPRESENTATIVE_VERTEX_ID, -1)), new Edge<LongWritable, FloatWritable>(getVertexId(), new FloatWritable(1.0f)));
			}

			if (getSuperstep() == 2) {
				// if the data vertex gets a message, create a centroid vertex with the data points of the data vertex
				while (msgIterator.hasNext()) {

					// Instantiate a new centroid vertex with vertex ID from representative node
					LongWritable vertexIndex = (LongWritable)msgIterator.next().get("R");
					MutableVertex<LongWritable, Text,
					FloatWritable, MapWritable> vertex = instantiateVertex();
					vertex.setVertexId(vertexIndex);

					// Clone datapoints of current data node to initial Centroid
					((KMeansVertex)vertex).setDatapoints(this.dataPoints);
					((KMeansVertex)vertex).setVertexValue(new Text("C"));

					try {
						LongWritable representativeId = new LongWritable(getConf().getLong(REPRESENTATIVE_VERTEX_ID, -1));
						addVertexRequest(vertex);
						// Add edge between centroid and representative vertex
						// C to R
						addEdgeRequest(vertexIndex,
								new Edge<LongWritable, FloatWritable>(
										representativeId, new FloatWritable(3.0f)));
						// R to C
						addEdgeRequest(representativeId,
								new Edge<LongWritable, FloatWritable>(
										vertexIndex, new FloatWritable(2.0f)));

						// send the data points of the newly created centroid to representative
						MapWritable msg = new MapWritable();
						msg.put(new Text("C"), new ArrayWritable(FloatWritable.class, this.dataPoints));
						msg.put(new Text("vertexId"), vertexIndex);
						sendMsg(representativeId, msg);

					} catch (IOException e) {
						e.printStackTrace();
					}


				}
			} 
			// On superstep 4, receive centroid coordinates from representative
			// and connect to the closest centroid
			else if (getSuperstep() == 4) {

				while (msgIterator.hasNext()) {

					MapWritable message = msgIterator.next();
					if (message.containsKey("R")) {
						MapWritable centroids = (MapWritable)message.get("R");
						float minimumDistance = Float.MAX_VALUE;
						LongWritable closestCentroidId = null;

						// compute the distance to each centroid and find the closest one
						for (Map.Entry<Writable,Writable> entry : centroids.entrySet()) {		
							float distance = computeDistance(this.dataPoints, (FloatWritable[])((ArrayWritable)entry.getValue()).get());
							if(distance < minimumDistance) {
								minimumDistance = distance;
								closestCentroidId = (LongWritable)entry.getKey();
							}
						}
						// create an edge between the data node and its closest centroid
						try {
							// from D to C
							addEdge(closestCentroidId, new FloatWritable(2.0f));
							// send the data points to the closest centroid.
							MapWritable msg = new MapWritable();
							msg.put(new Text("D"), new ArrayWritable(FloatWritable.class, this.dataPoints));
							sendMsg(closestCentroidId, msg);
							// from C to D
							addEdgeRequest(closestCentroidId,
									new Edge<LongWritable, FloatWritable>(
											getVertexId(), new FloatWritable(1.0f)));
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}		
			}
			// get the new centroid coordinates from the closest centroid and
			// compute the current closest centroid
			else if (getSuperstep() >= 7 && getSuperstep() % 3 == 1) {
				while (msgIterator.hasNext()) {			
					MapWritable message = msgIterator.next();
					if (message.containsKey("C")) {
						MapWritable centroids = (MapWritable)message.get("C");
						LongWritable currentCentroidId = (LongWritable)message.get("vertexId");
						float minimumDistance = Float.MAX_VALUE;
						LongWritable closestCentroidId = null;

						// compute the distance to each centroid and find the closest one
						for (Map.Entry<Writable,Writable> entry : centroids.entrySet()) {		
							float distance = computeDistance(this.dataPoints, (FloatWritable[])((ArrayWritable)entry.getValue()).get());
							if(distance < minimumDistance) {
								minimumDistance = distance;
								closestCentroidId = (LongWritable)entry.getKey();
							}
						}
						// if the closest centroid has been changed, create an edge between the data node 
						// and its closest centroid, and then remove the current connection
						try {
							if(closestCentroidId.get() != currentCentroidId.get()) {
								// from D to C
								addEdge(closestCentroidId, new FloatWritable(2.0f));
								// from C to D
								addEdgeRequest(closestCentroidId,
										new Edge<LongWritable, FloatWritable>(
												getVertexId(), new FloatWritable(1.0f)));

								// remove the edge from D to current C
								removeEdge(currentCentroidId);
								// remove the edge from current C to D
								removeEdgeRequest(currentCentroidId, getVertexId());
							}

							// send the data points to the closest centroid.
							MapWritable msg = new MapWritable();
							msg.put(new Text("D"), new ArrayWritable(FloatWritable.class, this.dataPoints));
							sendMsg(closestCentroidId, msg);

						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}	
			}	
			else {
				voteToHalt();
			}
		}

		/*** Centroid Vertex Behavior ***/
		if (getVertexValue().toString() == "C") { 

			// On superstep 4, receive centroid coordinates from representative
			// and connect to all other centroids
			if (getSuperstep() == 4) {

				while (msgIterator.hasNext()) {

					MapWritable message = msgIterator.next();

					if (message.containsKey("R")) {
						MapWritable other_centroids = (MapWritable)message.get("R");
						for (Map.Entry<Writable,Writable> entry : 
							other_centroids.entrySet()) {

							if (getVertexId().get() != ((LongWritable)entry.getKey()).get()) {

								addEdge((LongWritable)entry.getKey(), 
										new FloatWritable(2.0f));

							}
						}
					}
				}
			}

			// If maximum superstep reached, vote to halt
			else if (getSuperstep() >= getConf().getInt(SUPERSTEP_COUNT, -1)) {
				voteToHalt();
			}

			// Starting at superstep 5, on every superstep 2 in mod 3
			// thereafter, the centroid will receive messages from data 
			// vertices and recomputes own coordinates to send to other 
			// centroids
			else if (getSuperstep() >= 5 && getSuperstep() % 3 == 2) {

				// Initialize outgoing coordinate message
				MapWritable coordinate_message = new MapWritable();
				coordinate_message.put(new Text("vertexId"), getVertexId());

				// Initialize container to store new centroid coordinates
				FloatWritable[] compute_centroid = new FloatWritable[this.dataPoints.length];

				// Initialize counter to track messages
				int data_vertex_counter = 0;

				// Receive messages from attached data vertices
				while (msgIterator.hasNext()) {

					// Obtain message
					MapWritable message = msgIterator.next();

					// If the message is from data vertex, add its data points to
					// new centroid coordinate array
					if (message.containsKey("D")) {

						// Retrieve array of coordinates from message
						FloatWritable[] current = (FloatWritable[]) ((ArrayWritable)message.get("D")).get();

						// Update current centroid with coordinates
						for (int i = 0; i < compute_centroid.length; i++) {
							compute_centroid[i] = 
								new FloatWritable(compute_centroid[i].get() + current[i].get());
						}

						// Update data vertex message counter
						data_vertex_counter++;
					}
				}

				// If messages received from data vertices, update and broadcast
				// new centroid coordinates. Otherwise, if counter is 0, just broadcast
				// existing data points.
				if (data_vertex_counter > 0) {
					// Compute average of coordinates in centroid
					for (int i = 0; i < compute_centroid.length; i++) {
						compute_centroid[i] = new FloatWritable(compute_centroid[i].get() / data_vertex_counter);
					}	

					// Update data points of this centroid
					setDatapoints(compute_centroid);
				}

				// Send data points of current centroid to neighboring centroids
				coordinate_message.put(getVertexValue(),
						new ArrayWritable(FloatWritable.class, this.dataPoints));
				sendMsgToAllEdges(coordinate_message);

			}

			// Starting at superstep 6, on every superstep multiple of 3
			// thereafter, the centroid will receive messages from other 
			// centroids, create a map of those coordinates, and send it 
			// as message to data vertices
			else if (getSuperstep() >= 6 && getSuperstep() % 3 == 0) {

				// Initialize map of centroid coordinates with own coordinates
				MapWritable centroid_message = new MapWritable();

				MapWritable centroid_coordinates = new MapWritable();
				centroid_coordinates.put(getVertexId(),
						new ArrayWritable(FloatWritable.class, this.dataPoints));

				// Send this centroid's vertexId for identification in message
				centroid_message.put(new Text("vertexId"), getVertexId());

				// Receive messages from other centroids of their coordinates
				while (msgIterator.hasNext()) {

					MapWritable message = msgIterator.next();

					// If the message is from another centroid,
					// add coordinates to outgoing coordinate list
					if (message.containsKey("C")) {

						centroid_coordinates.put(message.get("vertexId"), 
								message.get("C"));

					}
				}

				// Add list of centroid coordinates to outgoing message
				centroid_message.put(getVertexValue(), 
						centroid_coordinates);

				// Send message to attached vertices with all coordinates
				sendMsgToAllEdges(centroid_message);
			}

			// On supersteps 7, 10, 13 etc: do nothing.
			else if (getSuperstep() >= 7 && getSuperstep() % 3 == 1) {
				// pass
			}

		}

		/*** End Centroid Vertex Behavior ***/

	}

	@Override
	public int run(String[] args) throws Exception {
		Options options = new Options();
		options.addOption("h", "help", false, "Help");
		options.addOption("w",
				"workers",
				true,
		"Number of workers");
		options.addOption("i",
				"input",
				true,
		"Input file");
		options.addOption("o",
				"output",
				true,
		"Output file");  
		options.addOption("k",
				"clusters",
				true,
		"Number of cluster centers");


		HelpFormatter formatter = new HelpFormatter();
		if (args.length == 0) {
			formatter.printHelp(getClass().getName(), options, true);
			return 0;
		}
		CommandLineParser parser = new PosixParser();
		CommandLine cmd = parser.parse(options, args);
		if (cmd.hasOption('h')) {
			formatter.printHelp(getClass().getName(), options, true);
			return 0;
		}
		if (!cmd.hasOption('w')) {
			System.out.println("Need to choose the number of workers (-w)");
			return -1;
		}
		if (!cmd.hasOption('i')) {
			System.out.println("Need to set the input file (-i)");
			return -1;
		}
		if (!cmd.hasOption('o')) {
			System.out.println("Need to set the output file (-o)");
			return -1;
		}
		if (!cmd.hasOption('k')) {
			System.out.println("Need to choose the number of clusters (-k)");
			return -1;
		}

		int number_of_clusters = Integer.parseInt(cmd.getOptionValue('k'));
		int workers = Integer.parseInt(cmd.getOptionValue('w'));
		GiraphJob job = new GiraphJob(getConf(), getClass().getName());
		job.setVertexClass(getClass());
		job.setVertexInputFormatClass(KMeansDataVertexInputFormat.class);
		job.setVertexOutputFormatClass(KMeansVertexOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(cmd.getOptionValue('i')));
		FileOutputFormat.setOutputPath(job, new Path(cmd.getOptionValue('o')));
		job.setWorkerConfiguration(workers, workers, 100.0f);

		// 100 Supersteps is a temporary simplicity solution...
		job.getConfiguration().setInt(
				SUPERSTEP_COUNT,
				100);

		job.getConfiguration().setInt(NUMBER_OF_CLUSTERS, number_of_clusters);
		
		// the vertex id of the representative
		job.getConfiguration().setLong(REPRESENTATIVE_VERTEX_ID, -1L);

		if (job.run(true) == true) {
			return 0;
		} else {
			return -1;
		}
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new KMeansVertex(), args));
	}

	public static class KMeansDataVertexReader extends
	TextVertexReader<LongWritable, Text, FloatWritable, MapWritable> {

		/** Vertices read so far */
		private long verticesRead = 0;
		private long largestVertexId = Long.MIN_VALUE;

		public KMeansDataVertexReader(
				RecordReader<LongWritable, Text> lineRecordReader) {
			super(lineRecordReader);
		}
		
		@Override
        public boolean nextVertex() throws IOException, InterruptedException {
			return getRecordReader().nextKeyValue();
        }

		@Override
        public BasicVertex<LongWritable, Text, FloatWritable, MapWritable>
          getCurrentVertex() throws IOException, InterruptedException {
			
			BasicVertex<LongWritable, Text, FloatWritable, MapWritable> vertex = BspUtils
			.<LongWritable, Text, FloatWritable, MapWritable> createVertex(getContext()
					.getConfiguration());

			if(verticesRead == 0) {
				// create a representative vertex which will be connected to all the data vertices
				LongWritable vertexId = new LongWritable(getContext().getConfiguration().getLong(REPRESENTATIVE_VERTEX_ID, -1));
				Text vertexValue = new Text("R");
				FloatWritable[] points = null;
				((KMeansVertex)vertex).setDatapoints(points);
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				vertex.initialize(vertexId, vertexValue, edges, null);
				verticesRead++;
				return vertex;
			}

			Text line = getRecordReader().getCurrentValue();
			try {
				// 
				StringTokenizer tokenizer = new StringTokenizer(line.toString());
				LongWritable vertexId = new LongWritable(Long.parseLong(tokenizer.nextToken()));
				Text vertexValue = new Text("D");
				FloatWritable[] points = new FloatWritable[tokenizer.countTokens()];

				// Read the data coordinates
				int counter = 0;
				while (tokenizer.hasMoreTokens()) {
					points[counter++] = new FloatWritable(Float.parseFloat(tokenizer.nextToken()));
				}

				// Set the data coordinates
				((KMeansVertex)vertex).setDatapoints(points);
				
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				// from D to R
				edges.put(new LongWritable(getContext().getConfiguration().getLong(REPRESENTATIVE_VERTEX_ID, -1)), new FloatWritable(3.0f));
				
				vertex.initialize(vertexId, vertexValue, edges, null);
				verticesRead++;
				if(largestVertexId < vertexId.get()) {
					largestVertexId = vertexId.get();
				}

			} catch (Exception e) {
				throw new IllegalArgumentException(
						"next: Couldn't get vertex from line " + line, e);
			}
			return vertex;
		}
		
		@Override
        public void close() throws IOException {
            super.close();
            getContext().getConfiguration().setLong(LARGEST_VERTEX_ID, largestVertexId);
        }

	}

	public static class KMeansDataVertexInputFormat extends TextVertexInputFormat<LongWritable, Text, FloatWritable, MapWritable> {

		@Override
		public VertexReader<LongWritable, Text, FloatWritable, MapWritable> createVertexReader(
				InputSplit split, TaskAttemptContext context) throws IOException {
			return new KMeansDataVertexReader(textInputFormat.createRecordReader(split, context));
		}

	}

	public static class KMeansVertexWriter extends
	TextVertexWriter<LongWritable, Text, FloatWritable> {
		public KMeansVertexWriter(
				RecordWriter<Text, Text> lineRecordWriter) {
			super(lineRecordWriter);
		}

		@Override
		public void writeVertex(
				BasicVertex<LongWritable, Text, FloatWritable, ?> vertex)
		throws IOException, InterruptedException {
			KMeansVertex currentVertex = (KMeansVertex)vertex;
			String outputStr = currentVertex.getVertexId().toString() + " :";
			if(currentVertex.getVertexValue().toString() == "C") {
				// for each connected data vertex of a centroid, output
				for(Edge<LongWritable, FloatWritable> edge : currentVertex.destEdgeMap.values()) {
					// the edge goes to a data vertex if the edge value is 1.0f
					if(edge.getEdgeValue().get() == 1.0f) {
						outputStr += "\t" + edge.getDestVertexId().toString();
					}
				}
				getRecordWriter().write(new Text(outputStr), null);
			}
		}
	}

	public static class KMeansVertexOutputFormat extends
	TextVertexOutputFormat<LongWritable, Text, FloatWritable> {

		@Override
		public VertexWriter<LongWritable, Text, FloatWritable>
		createVertexWriter(TaskAttemptContext context)
		throws IOException, InterruptedException {
			RecordWriter<Text, Text> recordWriter = textOutputFormat.getRecordWriter(context);
			return new KMeansVertexWriter(recordWriter);
		}
	}



}