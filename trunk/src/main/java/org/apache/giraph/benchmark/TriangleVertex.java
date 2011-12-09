package org.apache.giraph.benchmark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
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
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.LongWritable;
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
// import org.apache.log4j.Logger;

import com.google.common.collect.Maps;





/**
 * @param <I> Vertex id
 * @param <V> Vertex value
 * @param <E> Edge value
 * @param <M> Message value
 */
public class TriangleVertex extends
Vertex<LongWritable, MapWritable, BooleanWritable, VertexPairWritable> 
implements Tool{
	
	private Configuration conf;
	
	// private static final Logger LOG = Logger.getLogger(TriangleVertex.class);
	
	@Override
	public Configuration getConf() {
		return conf;
	}

	@Override
	public void setConf(Configuration conf) {
		this.conf = conf;
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
		
        int workers = Integer.parseInt(cmd.getOptionValue('w'));
        GiraphJob job = new GiraphJob(getConf(), getClass().getName());
        job.setVertexClass(TriangleVertex.class);
        job.setVertexInputFormatClass(TriangleVertexInputFormat.class);
		job.setVertexOutputFormatClass(TriangleVertexOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(cmd.getOptionValue('i')));
		FileOutputFormat.setOutputPath(job, new Path(cmd.getOptionValue('o')));
        job.setWorkerConfiguration(workers, workers, 100.0f);

        if (job.run(true) == true) {
            return 0;
        } else {
            return -1;
        }
	}
	
	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new TriangleVertex(), args));
	}
	
	@Override
	public void compute(Iterator<VertexPairWritable> msgIterator)
			throws IOException {

		// send message to all of the vertices whose vertexId is greater then the current one
		if (getSuperstep() == 0) {
			Iterator<LongWritable> edgeIterator = iterator();
			ArrayList<LongWritable> firstVertexList = new ArrayList<LongWritable>();
			boolean msgReady = false;
			VertexPairWritable msg = new VertexPairWritable();
			while(edgeIterator.hasNext()) {
				LongWritable targetVertexId = edgeIterator.next();
				if( getVertexId().get() > targetVertexId.get() ) {
					firstVertexList.add(targetVertexId);
				}
				else if(firstVertexList.size() > 0) {
					if( getVertexId().get() < targetVertexId.get() ) {
						if(!msgReady) {
							LongWritable[] firstVertexArray = new LongWritable[1];
							firstVertexArray = firstVertexList.toArray(firstVertexArray);
							LongArrayWritable firstVertexArrayWritable = new LongArrayWritable();
							firstVertexArrayWritable.set(firstVertexArray);
							msg.set(getVertexId(), firstVertexArrayWritable);
							msgReady = true;
						}
						sendMsg(targetVertexId, msg);	
					}
				}
				// if there is no message to send, stop iterating
				else {
					voteToHalt();
					break;
				}
			}
		}
		// receive the messages and check if there is an edge from the current vertex to the first vertexId in the incoming message
		else if (getSuperstep() == 1) {
			Hashtable<LongWritable, ArrayList<LongWritable>> triangleList = new Hashtable<LongWritable, ArrayList<LongWritable>>();
			while (msgIterator.hasNext()) {
				VertexPairWritable msg = msgIterator.next();
				LongWritable[] list = (LongWritable[]) msg.getFirstVertexList().toArray();
				for(LongWritable firstVertexId : list) {
					if(triangleList.containsKey(firstVertexId)) {
						ArrayList<LongWritable> newList = triangleList.get(firstVertexId);
						newList.add(msg.getMiddleVertex());
						triangleList.put(firstVertexId, newList);
					}
					else {
						if(hasEdge(firstVertexId)) {
							ArrayList<LongWritable> newList = new ArrayList<LongWritable>();
							newList.add(msg.getMiddleVertex());
							triangleList.put(firstVertexId, newList);						
						}
					}
				}
			}
			// set the vertex value
			MapWritable vertexValue = new MapWritable();
			for(Entry<LongWritable, ArrayList<LongWritable>> entry: triangleList.entrySet()) {
				
				LongWritable[] valueArray = new LongWritable[1];
				valueArray = entry.getValue().toArray(valueArray);
				LongArrayWritable valueArrayWritable = new LongArrayWritable();
				valueArrayWritable.set(valueArray);
				vertexValue.put(entry.getKey(), valueArrayWritable);		
			}
			setVertexValue(vertexValue);
		}

		voteToHalt();

		
	}
	
	public static class TriangleVertexInputFormat extends TextVertexInputFormat<LongWritable,  MapWritable, BooleanWritable, VertexPairWritable> {

		@Override
		public VertexReader<LongWritable, MapWritable, BooleanWritable, VertexPairWritable> createVertexReader(
				InputSplit split, TaskAttemptContext context)
				throws IOException {
			return new TriangleVertexReader(textInputFormat.createRecordReader(split, context));
		}
		
	}
	
	public static class TriangleVertexReader extends TextVertexReader<LongWritable,  MapWritable, BooleanWritable, VertexPairWritable> {

		public TriangleVertexReader(
				RecordReader<LongWritable, Text> lineRecordReader) {
			super(lineRecordReader);
		}
		
		@Override
		public BasicVertex<LongWritable, MapWritable, BooleanWritable, VertexPairWritable> getCurrentVertex()
				throws IOException, InterruptedException {
			BasicVertex<LongWritable, MapWritable, BooleanWritable, VertexPairWritable> vertex = BspUtils.<LongWritable, MapWritable, BooleanWritable, VertexPairWritable>createVertex(getContext().getConfiguration());
			
			Text line = getRecordReader().getCurrentValue();
			try {
				StringTokenizer tokenizer = new StringTokenizer(line.toString());
		
				LongWritable vertexId = new LongWritable(Long.parseLong(tokenizer.nextToken()));
				MapWritable vertexValue = new MapWritable();
				
				Map<LongWritable, BooleanWritable> edges = Maps.newHashMap();
				while (tokenizer.hasMoreTokens()) {
					LongWritable destVertexId = new LongWritable(Long.parseLong(tokenizer.nextToken()));
					edges.put(destVertexId, new BooleanWritable(false));
				}
				vertex.initialize(vertexId, vertexValue, edges, null);
			} catch (Exception e) {
				throw new IllegalArgumentException(
						"next: Couldn't get vertex from line " + line, e);
			}
			
			return vertex;
		}

		@Override
		public boolean nextVertex() throws IOException, InterruptedException {
			return getRecordReader().nextKeyValue();
		}
	
	}
	
	public static class TriangleVertexWriter extends
    TextVertexWriter<LongWritable, MapWritable, BooleanWritable> {
		public TriangleVertexWriter(
		        RecordWriter<Text, Text> lineRecordWriter) {
		    super(lineRecordWriter);
		}
		
		@Override
		public void writeVertex(
		        BasicVertex<LongWritable, MapWritable, BooleanWritable, ?> vertex)
		        throws IOException, InterruptedException {
			
			//LOG.info("Started vertex: " + vertex.getVertexId().toString() + ", Value Size: " + vertex.getVertexValue().size());
			for(Entry<Writable, Writable> entry : vertex.getVertexValue().entrySet()) {
				for(Writable middleVertexId : ((LongArrayWritable)entry.getValue()).get()) {
					String output = entry.getKey().toString() + " - " + middleVertexId.toString() + " - " + vertex.getVertexId().toString();
					getRecordWriter().write(new Text(output), null);
				}
			}
		}
	}
		
	public static class TriangleVertexOutputFormat extends
	TextVertexOutputFormat<LongWritable, MapWritable, BooleanWritable> {
		
		@Override
		public VertexWriter<LongWritable, MapWritable, BooleanWritable>
		    createVertexWriter(TaskAttemptContext context)
		        throws IOException, InterruptedException {
		    RecordWriter<Text, Text> recordWriter =
		        textOutputFormat.getRecordWriter(context);
		    return new TriangleVertexWriter(recordWriter);
		}
	}
	
}


