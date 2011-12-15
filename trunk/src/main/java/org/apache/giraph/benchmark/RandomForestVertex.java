package org.apache.giraph.benchmark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Map.Entry;

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
 * Random Forest: a collection of unpruned decision trees
 * 
 * Type Parameters
 * I - vertex ID
 * V - vertex data
 * E - edge data
 * M - message data
 */
public class RandomForestVertex extends
Vertex<LongWritable, Text, FloatWritable, MapWritable>
implements Tool {

	private MapWritable forestData; 
	private RandomForestNodeWritableAdaptation tree;
	
	public void setData(MapWritable data) {
		this.forestData = data;
	}
	
	public MapWritable getData() {
		return this.forestData;
	}
	
	/* How many attributes there are in our dataset */
	public static String ATTRIBUTE_COUNT = "RandomForestVertex.attributeCount";

	/* How many classification trees to create */
	public static String FOREST_SIZE = "RandomForestVertex.forestSize";
	
	/* Size of training data set */
	public static String TRAINING_DATA_CASES = "RandomForestVertex.trainingDataCases";
	
	/* Vertex ID of the representative vertex */
	public static String REPRESENTATIVE_VERTEX_ID = "RandomForestVertex.representativeVertexId";
	
	@Override
	public void compute(Iterator<MapWritable> msgIterator) throws IOException {
		
		/* Start representative vertex behavior */
		if (getVertexValue().toString() == "R") { 
	
			// On superstep 0, create Forest Vertices
			if (getSuperstep() == 0) {
				
				// Get quantity of existing vertices, quantity of trees to build
				long forest_vertex_id = getNumVertices() + 2;
				int trees_to_build = getConf().getInt(FOREST_SIZE, -1);
				
				// Initialize test result tracker
				forestData.put(new Text("expected_classification"), new MapWritable());
				forestData.put(new Text("forest_classification"), new MapWritable());
				
				// Build random forest by creating vertices
				for (long i = forest_vertex_id; i < trees_to_build; i++) {
					
					// Create forest vertex
					LongWritable vertexIndex = new LongWritable(i);
					MutableVertex<LongWritable, Text, FloatWritable, MapWritable> tree = 
						instantiateVertex();
					tree.setVertexId(vertexIndex);
					
					MapWritable forestData = new MapWritable();
					((RandomForestVertex)tree).setData(forestData);
					((RandomForestVertex)tree).setVertexValue(new Text("C"));
					
					try {
					// Create edge from representative node to tree, and back
					addVertexRequest(tree);
					addEdgeRequest(vertexIndex,
							new Edge<LongWritable, FloatWritable>(
									new LongWritable(-1L), new FloatWritable(1.0f)));
					addEdgeRequest(new LongWritable(-1L),
							new Edge<LongWritable, FloatWritable>(
									vertexIndex, new FloatWritable(1.0f)));
				
					} catch (IOException e) {
							e.printStackTrace();		
					}
				}
			}
			
			// On superstep 1, receive, combine, and pass on training data
			else if (getSuperstep() == 1) {
				
				// Receive training data
				MapWritable training_data = new MapWritable();
				while (msgIterator.hasNext()) {
					
					// If coming message from a vertex of type TR, 
					// add to training_data message
					MapWritable message = msgIterator.next();
					
					if (message.containsKey("TR")) {
						
						training_data.put((LongWritable)message.get("vertexId"), 
								(ArrayWritable)message.get("TR"));
					}
				}
				
				// Send training data to classification forest
				MapWritable message_training_data = new MapWritable();
				message_training_data.put(getVertexValue(), training_data);
				this.sendMsgToAllEdges(message_training_data);
			}
			
			// On superstep 2, receive, combine, and pass on testing data
			else if (getSuperstep() == 2) {
				
				// Testing data
				MapWritable testing_data = new MapWritable();
				while (msgIterator.hasNext()) {
					
					// If coming message from a vertex of type TR, 
					// add to training_data message
					MapWritable message = msgIterator.next();
					
					if (message.containsKey(new Text("TE"))) {
						
						// Add testing data to outgoing message container
						testing_data.put((LongWritable)message.get("vertexId"), 
								((ArrayWritable)message.get("TE")));
					
						// Add expected result of test to tracker
						((MapWritable)forestData.get("expected_classification")
								).put((LongWritable)message.get("vertexId"), 
										((ArrayWritable)message.get(new Text("TE"))).get()[0]);
					}
				}
				
				// Send training data to classification forest
				MapWritable message_testing_data = new MapWritable();
				message_testing_data.put(getVertexValue(), testing_data);
				this.sendMsgToAllEdges(message_testing_data);
			}
			
			// On superstep 4, receive and tally votes from forest 
			else if (getSuperstep() == 4) {
				
				// the tracker is a MapWritable in forestData of the representative
				// which is used to tally votes from the forest. It should have an entry for
				// each test datapoint, which will serve to track attributes. Once
				// tally is complete, compute percentage who voted for expected classification
				// from test data
				
				while (msgIterator.hasNext()) {
					
					// Looking for messages with classification data
					MapWritable message = msgIterator.next();	
				
					if (message.containsKey(new Text("tree_result"))) {
						
						LongWritable test_id = (LongWritable)message.get(new Text("test_vertex_id"));
						FloatWritable classification = (FloatWritable)message.get(new Text("tree_result"));
						
						// Will receive a floatwritable attribute as a classification
						// Check if this attribute value for the given test vertex
						// has already been voted...if not, create it
						
						if (!((MapWritable)(
								(MapWritable)
								forestData.get(new Text("forest_classification")))
								.get(test_id)).containsKey(classification)) {
							
							((MapWritable)forestData.get(new Text("forest_classification"))
									).put(classification, new IntWritable(1));
						}
						else
						{
							// Update its vote counter
							int current_votes = 
								((IntWritable) ((MapWritable)
								((MapWritable)forestData
										.get(new Text("forest_classification"))).get(test_id))
										.get(classification))
										.get();
						
							((MapWritable)
									((MapWritable)forestData
											.get(new Text("forest_classification")))
											.get(test_id)).put(classification,
													new IntWritable(current_votes + 1));
							
							// 		// Add expected result of test to tracker
							((MapWritable)forestData.get("forest_classification")
							).put((LongWritable)message.get("vertexId"), 
									((ArrayWritable)message.get(new Text("TE"))).get()[0]);
						}
					}
				}
				
				// Compute percentage correct classification
				MapWritable expected = (MapWritable)forestData.get(new Text("expected_classification"));
				MapWritable classifications = (MapWritable)forestData.get(new Text("forest_classification"));
				Float accuracy = 0.0f;
				
				// For each tally in forest classifications, updated accuracy if the expected
				// result got the most votes
				for (Entry<Writable, Writable> t_entry : expected.entrySet()) {
					
					// Get actual values of classifications and vote counts
					FloatWritable res = (FloatWritable) expected.get(t_entry.getKey());
					Map<Float, Integer> votes = new HashMap<Float,Integer>();
					for (Entry<Writable, Writable> v_entry : 
						((MapWritable)classifications.get(t_entry.getKey())).entrySet()) {
						votes.put(((FloatWritable)v_entry.getKey()).get(), 
								((IntWritable)v_entry.getValue()).get());
					}
					
					// Compute prevalance
					if (Collections.max(votes.values()) == votes.get(res)) {
						accuracy += 1.0f;
					}
				}
				
				forestData.put(new Text("accuracy"), new FloatWritable(accuracy/expected.size()));
				
				// Once tally complete, halt algorithm
				voteToHalt();
			}
		}
		/* End representative vertex behavior */
		
		/* Start tree classifier vertex behavior */
		if (getVertexValue().toString() == "C") {
		
			// The tree comes into existence on superstep 1, and simply
			// initializes a RandomForestTreeNode inside of itself.
			// This will later be trained with training data and then used for
			// classification with test data
			if (getSuperstep() == 1) {
				
				// Get quantity of attributes for consideration
				int attribute_count = getConf().getInt(ATTRIBUTE_COUNT, -1);
				
				// Initialize internal tree
				tree = new RandomForestNodeWritableAdaptation(attribute_count);
			}
			
			// Training Behavior: Superstep 2
			// Receive training data, choose subset, and run training algorithm
			if (getSuperstep() == 2) {
			
				ArrayList<ArrayWritable> training = new ArrayList<ArrayWritable>();
				
				// Receive message with training data, copy to local training container
				while (msgIterator.hasNext()) {
				
					MapWritable message = new MapWritable();
					
					if (message.containsKey(new Text("R"))) {
						
						MapWritable training_map = (MapWritable)message.get(new Text("R"));
						for (Entry<Writable,Writable> train : training_map.entrySet()) {
							
							// Add only data to training data. Doesn't matter which 
							// vertex ID it comes from.
							training.add((ArrayWritable)train.getValue());
						}
						
					}
					
				}
				
				// Select a random subset of data, with replacement
				ArrayList<ArrayWritable> training_data_subset = new ArrayList<ArrayWritable>();
				Random r = new Random();
				
				// Random + replacement training data is same size as actual training
				// data set, just with slightly different members.
				while (training_data_subset.size() < training.size()) {
					
					// Add random members of training data until subset is of
					// equal size
					training_data_subset.add(training.get(r.nextInt(training.size())));
				
				}
			
				// Train tree with subset of data
				tree.train(training_data_subset, new IntWritable(0));
			}
			
			// Testing Behavior: Superstep 3
			// Receive testing data and classify, sending a results message
			// to representative vertex for voting.
			// Once no testing data is left, vote to halt.
			if (getSuperstep() == 3) {
				
				// Initialize list of entries to classify
				MapWritable classification_queue = new MapWritable();
				
				// Receive testing data from vertex
				while (msgIterator.hasNext() && classification_queue.size() == 0) {
				
					MapWritable message = new MapWritable();
					
					if (message.containsKey(new Text("R"))) {
						
						classification_queue = (MapWritable)message.get(new Text("R"));
						
					}
					
				}
				
				// Classify each sample, and send result back to representative
				for (Entry<Writable, Writable> entry : classification_queue.entrySet()) {
					
					// Create result message and fill in with classification result
					MapWritable result_message = new MapWritable();
					
					FloatWritable result = tree.classify((ArrayWritable)entry.getValue());
					result_message.put(new Text("test_vertex_id"), (LongWritable)entry.getKey());
					result_message.put(new Text("tree_result"), result);
					
					// Send message with classification to root
					sendMsg(new LongWritable(-1L), result_message);
				}
				
				// Once classification is complete, vote to halt.
				voteToHalt();
			}
		
		}
		/* End classifier vertex behavior */
		
		/* Start training vertex behavior */
		if (getVertexValue().toString() == "TR") {
			
			// On superstep 0, set node to be a Test node if its id is greater than
			// the training threshold
			if (this.getSuperstep() == 0) {
			
				// First task is to specify own type as vertex with either
				// training or testing, based on position in input data set
				int trainingDataSize = getConf().getInt(TRAINING_DATA_CASES, -1);
				
				// Send message to representative node with data
				if (getVertexId().get() <= trainingDataSize) {
					
					MapWritable trainingData = new MapWritable();
					
					trainingData.put(new Text("TR"), 
							(ArrayWritable)forestData.get(new Text("data")));
					
					sendMsg(new LongWritable(-1L), trainingData);
					
					// Training data no longer needed 
					voteToHalt();
				} 
				
				// Otherwise, designate as a testing vertex
				else {
					
					setVertexValue(new Text("TE"));
					
				}
			}
	
		}
		/* End training vertex behavior */

		/* Start testing vertex behavior */
		if (getVertexValue().toString() == "TE") {
	
			// Test vertices only exist after they are designated in superstep 0,
			// so this vertex is only active on superstep 1 when it sends its 
			// data to the representative vertex. Then it votes to halt, no 
			// further work is needed. 
			if (getSuperstep() == 1) {
			
				// Create message with testing data
				MapWritable testingData = new MapWritable();
				
				testingData.put(new Text("TE"), 
						(ArrayWritable)forestData.get(new Text("data")));
				
				// Sent training data to representative vertex
				sendMsg(new LongWritable(-1L), testingData);
				
				// Action complete, vote to halt execution
				voteToHalt();
			}
		}
		/* End testing vertex behavior */
		
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
				"input file",
				true,
		"Input data file");
		options.addOption("o",
				"output",
				true,
		"Output file");
		options.addOption("n",
				"training cases",
				true,
		"Number of training cases");
		options.addOption("m",
				"attributes",
				true,
		"Number of classification attributes");
		options.addOption("f",
				"trees",
				true,
		"Number of trees in forest");
		
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
			System.out.println("Need to set the input training file (-i)");
			return -1;
		}
		if (!cmd.hasOption('n')) {
			System.out.println("Need to set the number of training cases (-n)");
			return -1;
		}
		if (!cmd.hasOption('o')) {
			System.out.println("Need to set the output file (-o)");
			return -1;
		}
		if (!cmd.hasOption('m')) {
			System.out.println("Need to set the number of attributes (-m)");
			return -1;
		}
		if (!cmd.hasOption('f')) {
			System.out.println("Need to choose the size of the forest (-f)");
			return -1;
		}

		// Obtain parameters for forest construction
		int number_of_attributes = Integer.parseInt(cmd.getOptionValue('m'));
		int forest_size = Integer.parseInt(cmd.getOptionValue('f'));
		int training_cases = Integer.parseInt(cmd.getOptionValue('n'));
		int workers = Integer.parseInt(cmd.getOptionValue('w'));
		
		// Create giraph job
		GiraphJob job = new GiraphJob(getConf(), getClass().getName());
		job.setVertexClass(getClass());
		job.setVertexInputFormatClass(RandomForestVertexInputFormat.class);
		job.setVertexOutputFormatClass(RandomForestVertexOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(cmd.getOptionValue('i')));
		FileOutputFormat.setOutputPath(job, new Path(cmd.getOptionValue('o')));
		job.setWorkerConfiguration(workers, workers, 100.0f);

		// Set Random Forest Job Options
		job.getConfiguration().setInt(ATTRIBUTE_COUNT, number_of_attributes);
		job.getConfiguration().setInt(FOREST_SIZE, forest_size);
		job.getConfiguration().setInt(TRAINING_DATA_CASES, training_cases);
		job.getConfiguration().setLong(REPRESENTATIVE_VERTEX_ID, -1L);

		if (job.run(true) == true) {
			return 0;
		} else {
			return -1;
		}
	}
	
	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new RandomForestVertex(), args));
	}
	
	/**
	 * Writable Compatible refactor of RandomForestClassificationTree
	 */
	/* to refactor from RandomForestClassificationTree */
	public static class RandomForestNodeWritableAdaptation {
		
		private DoubleWritable attributeSubsetSize;
		private MapWritable decisionTree;
		
		// Constructor
		public RandomForestNodeWritableAdaptation(int quantity_attributes) {
			
			// Initialize attribute subset size, used for splitting
			this.attributeSubsetSize = new DoubleWritable(Math.ceil(Math.sqrt(quantity_attributes)));
			
			// Initialize internal decision tree
			decisionTree = new MapWritable();
			
		}
		
		// Training Initializer
		public void train(ArrayList<ArrayWritable> training_data,
				IntWritable target_attribute) {
			
			// FIXME 
		}
		
		// Classifier
		public FloatWritable classify (ArrayWritable to_classify) {
			
			// FIXME
			
			return new FloatWritable(0.0f);
		}
		
	}
	
	/**
	 * I/O Formats, Reader, Writer
	 */
	
	public static class RandomForestVertexReader extends
	TextVertexReader<LongWritable, Text, FloatWritable, MapWritable> {
		
		/* Vertices read so far */
		private long verticesRead = 0;
	
		public RandomForestVertexReader(
				RecordReader<LongWritable, Text> lineRecordReader) {
			super(lineRecordReader);
		}
	
		@Override
        public boolean nextVertex() throws IOException, InterruptedException {
			return getRecordReader().nextKeyValue();
        }

		public BasicVertex<LongWritable, Text, FloatWritable, MapWritable>
		getCurrentVertex() throws IOException, InterruptedException {
		
			BasicVertex<LongWritable, Text, FloatWritable, MapWritable> vertex = BspUtils
			.<LongWritable, Text, FloatWritable, MapWritable> createVertex(getContext()
					.getConfiguration());
			
			if(verticesRead == 0) {
				
				LongWritable vertexId = new LongWritable(getContext().getConfiguration().getLong(REPRESENTATIVE_VERTEX_ID, -1));
				Text vertexValue = new Text("R");
				
				MapWritable representative = new MapWritable();
				((RandomForestVertex)vertex).setData(representative);
				
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				
				vertex.initialize(vertexId, vertexValue, edges, null);
				verticesRead++;
				return vertex;
			}
		
			Text line = getRecordReader().getCurrentValue();
			try {
				
				// Read data tokens
				StringTokenizer tokenizer = new StringTokenizer(line.toString(), ",");
				LongWritable vertexId = new LongWritable(new Long(verticesRead));
				
				// Initialize data vertex with data
				Text vertexValue = new Text("TR");
				
				MapWritable dataVertex = new MapWritable();
				
				FloatWritable[] attributes = new FloatWritable[tokenizer.countTokens()];
				int counter = 0;
				while (tokenizer.hasMoreTokens()) {
					attributes[counter++] = new FloatWritable(Float.parseFloat(tokenizer.nextToken()));
				}
				
				dataVertex.put(new Text("data"), new ArrayWritable(FloatWritable.class, attributes));
				((RandomForestVertex)vertex).setData(dataVertex);
				
				// Create edge map
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				edges.put(new LongWritable(getContext().getConfiguration().getLong(REPRESENTATIVE_VERTEX_ID, -1)), new FloatWritable(3.0f));
				
				vertex.initialize(vertexId, vertexValue, edges, null);
				verticesRead++;
				
			} catch (Exception e) {
				throw new IllegalArgumentException(
						"next: Couldn't get vertex from line " + line, e);
			}
			return vertex;
		
		}
		
		@Override
		public void close() throws IOException {
			super.close();
		}

	}
	
	public static class RandomForestVertexInputFormat extends
	TextVertexInputFormat<LongWritable, Text, FloatWritable, MapWritable> {
		
		@Override
		public VertexReader<LongWritable, Text, FloatWritable, MapWritable> 
		createVertexReader(InputSplit split, TaskAttemptContext context) 
		throws IOException {
			
			return new RandomForestVertexReader(textInputFormat.createRecordReader(split, context));
		}
	}
	
	public static class RandomForestVertexOutputFormat extends
	TextVertexOutputFormat<LongWritable, Text, FloatWritable>{

		@Override
		public VertexWriter<LongWritable, Text, FloatWritable> 
		createVertexWriter(TaskAttemptContext context) throws IOException,
				InterruptedException {
			RecordWriter<Text, Text> recordWriter = textOutputFormat.getRecordWriter(context);
			return new RandomForestVertexWriter(recordWriter);
		}

	}
	
	public static class RandomForestVertexWriter extends
	TextVertexWriter<LongWritable, Text, FloatWritable> {

		public RandomForestVertexWriter(
				RecordWriter<Text, Text> lineRecordWriter) {
			super(lineRecordWriter);
		}

		@Override
		public void writeVertex(
				BasicVertex<LongWritable, Text, FloatWritable, ?> vertex)
		throws IOException, InterruptedException {

			// Only record current vertex if it is a tree node, vs a training/testing set node
			if (vertex.getVertexValue().toString() == "R") {
				
				Float accuracy = ((FloatWritable)((RandomForestVertex)vertex).getData().get(new Text("accuracy"))).get();
				
				Text output = new Text("Percent Accuracy: " + accuracy*100 );
				getRecordWriter().write(new Text(output), null);
				
			}
		}
	}

}
