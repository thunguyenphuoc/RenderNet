import tensorflow as tf

OUTPUT_NODE_NAME = ["encoder/output"]
OUTPUT_GRAPH_NAME = "./model/3d2d_render.pb"
with tf.Session() as sess:
  print("starting")
  new_saver = tf.train.import_meta_graph("./model/3d2d_renderer.meta")
  new_saver.restore(sess, "./model/3d2d_renderer")
  print("The pre-trained model has been loaded")

  output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    tf.get_default_graph().as_graph_def(),
    OUTPUT_NODE_NAME
  )

  with tf.gfile.GFile(OUTPUT_GRAPH_NAME, "wb") as f:
      f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))