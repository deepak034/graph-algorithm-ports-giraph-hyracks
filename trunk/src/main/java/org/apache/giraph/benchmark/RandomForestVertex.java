package org.apache.giraph.benchmark;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.giraph.graph.BasicVertex;
import org.apache.giraph.graph.BspUtils;
import org.apache.giraph.graph.GiraphJob;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.graph.VertexReader;
import org.apache.giraph.graph.VertexWriter;
import org.apache.giraph.lib.TextVertexInputFormat;
import org.apache.giraph.lib.TextVertexOutputFormat;
import org.apache.giraph.lib.TextVertexInputFormat.TextVertexReader;
import org.apache.giraph.lib.TextVertexOutputFormat.TextVertexWriter;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
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
	
	public void setData(MapWritable data) {
		this.forestData = data;
	}
	
	public static class RandomForestWritableAdaptation {
		
		/* Constructor adapter */
		public RandomForestWritableAdaptation() {}
		
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
				MapWritable tracker = new MapWritable();
				this.forestData.put(new Text("expected_classification"), tracker);
				
				// Build random forest by creating vertices
				for (long i = forest_vertex_id; i < trees_to_build; i++) {
					
					// Create forest vertex
					
					
					// Create link from representative node to vertex
					
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
					
					if (message.containsKey("TE")) {
						
						// Add testing data to outgoing message container
						testing_data.put((LongWritable)message.get("vertexId"), 
								((ArrayWritable)message.get("TE")));
					
						// Add expected result of test to tracker
						((MapWritable)forestData.get("expected_classification")).put((LongWritable)message.get("vertexId"), 
								((ArrayWritable)message.get("TE")).get()[0]);
					}
				}
				
				// Send training data to classification forest
				MapWritable message_testing_data = new MapWritable();
				message_testing_data.put(getVertexValue(), testing_data);
				this.sendMsgToAllEdges(message_testing_data);
			}
			
			// On superstep 4, receive and tally votes from forest 
			else if (getSuperstep() == 4) {
				
				// Once tally complete, halt algorithm
				voteToHalt();
			}
		}
		/* End representative vertex behavior */
		
		/* Start tree classifier vertex behavior */
		if (getVertexValue().toString() == "C") {
		
			// The tree comes into existence on superstep 1, 
			// so nothing happens until SS2 when training begins.
			
			// Training Behavior: Superstep 2
			// Receive training data and run training algorithm
			if (getSuperstep() == 2) {
				
			}
			
			// Testing Behavior: Superstep 3
			// Receive testing data and classify, sending a results message
			// to representative vertex for voting.
			// Once no testing data is left, vote to halt.
			if (getSuperstep() == 3) {
				
				// Once classification is complete, vote to halt.
				voteToHalt();
			}
		
		}
		/* End classifier vertex behavior */
		
		/* Start training vertex behavior */
		if (getVertexValue().toString() == "TR") {
			
			voteToHalt();
		}
		/* End training vertex behavior */

		/* Start testing vertex behavior */
		if (getVertexValue().toString() == "TE") {
	
			voteToHalt();
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
				
				//Double accuracy = ((DoubleWritable) vertex.getVertexValue().get("accuracy")).get();
				
				Text output = new Text("Just a test output");
				getRecordWriter().write(new Text(output), null);
				
			}
		}
	}

}
