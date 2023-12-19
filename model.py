# -*- coding: utf-8 -*-

"""Contains definitions for CompSegNet model

Ref: Mohamed Traoré, Emrah Hancer, Refik Samet, Zeynep Yıldırım, Nooshin Nemati,
     “CompSegNet: An enhanced U-shaped architecture for nuclei segmentation in H&E
     histopathology images,” Biomedical Signal Processing and Control, under revision, 
     2023.
      
Affil.:
     * CompSeg Lab, Breast Cancer Research Group
     * Ankara University, Ankara, Turkey
     * Website: http://compseg.ankara.edu.tr/
     * Email: compseg@ankara.edu.tr
"""


import tensorflow as tf                                                        
from keras_nlp.layers import TransformerEncoder

class IGCBlock(tf.keras.layers.Layer):
    """Implements the Improved Global Context Block.

    Args:
        channels (int): Number of output channels.
        ratio (int): Ratio used in computing the hidden feature size (default is 4).
        padding (str): Padding type for convolutional layers (default is 'same').
        **kwargs: Additional keyword arguments for the parent class.
                
    Attributes:
        hf_size (int): Hidden feature size used in context transform.
        hw_flatten (Lambda): Lambda layer for flattening spatial dimensions.
        transpose (Lambda): Lambda layer for transposing dimensions.
        add (Add): Keras Add layer for element-wise addition.
        relu (ReLU): Keras ReLU activation layer.
        softmax (Softmax): Keras Softmax activation layer.
        ln (LayerNormalization): Keras LayerNormalization layer.
        conv1 (Conv2D): Value generator for context modeling.
        conv2 (Conv2D): Keys generator for context mask.
        conv3 (Conv2D): Hidden features generator for context transform.
        conv4 (Conv2D): Hidden features projector for context transform.

    Methods:
        call(x, verbose=False): Perform the forward pass for the IGCBlock.

    """
    
    def __init__(self, channels, ratio=4, padding='same', **kwargs):
        """Initialize IGCBlock."""
        super(IGCBlock, self).__init__(**kwargs)
        self.hf_size = max(channels // ratio, 32)
        self.hw_flatten = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])
        )        
        self.transpose = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1])
        )
        
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.ReLU()
        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.ln = tf.keras.layers.LayerNormalization()
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters=channels, kernel_size=1, strides=1, padding=padding
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1, strides=1, padding=padding
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.hf_size, kernel_size=1, strides=1, padding=padding
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=channels, kernel_size=1, strides=1, padding=padding
        )    
    
    def call(self, x, verbose=False):
        """Forward pass for the IGCBlock."""
        inputs = x
        inputs = self.conv1(inputs)
        inputs = self.hw_flatten(inputs)
        inputs = self.transpose(inputs)
        inputs = tf.expand_dims(inputs, axis=1)
        
        # Generate context mask
        context_mask = self.conv2(x)
        context_mask = self.hw_flatten(context_mask)
        context_mask = self.softmax(context_mask)
        context_mask = self.transpose(context_mask)
        context_mask = tf.expand_dims(context_mask, axis=-1)
        
        # Context Modeling
        context = tf.matmul(inputs, context_mask)
        context = tf.reshape(context, shape=[tf.shape(x)[0], 1, 1, tf.shape(x)[-1]])
        
        # Context transform
        context_transform = self.conv3(context)
        context_transform = self.ln(context_transform)
        context_transform = self.relu(context_transform)
        context_transform = self.conv4(context_transform)

        return self.add([x, context_transform])


class CSegBlock(tf.keras.layers.Layer):
    """Implements the CompSeg (CSeg) block.

    Agrs:
        t (int): Width multiplier for filter expansion.
        filters (int): Number of input filters.
        kernel_size (int or tuple): Size of the depthwise convolution kernel.
        strides (int or tuple): Strides for the depthwise convolution.
        out_channels (int): Number of output filters for the projection convolution.
        padding (str): Padding type for convolutional layers (default is 'same').
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        expansion_conv (Conv2D): Expansion convolutional layer.
        expansion_bn (BatchNormalization): Batch normalization for the expansion convolution.
        dwise_conv (DepthwiseConv2D): Depthwise convolutional layer.
        dwise_bn (BatchNormalization): Batch normalization for the depthwise convolution.
        upsample (UpSampling2D): Upsampling layer using bilinear interpolation.
        projection_conv (Conv2D): Projection convolutional layer.
        projection_bn (BatchNormalization): Batch normalization for the projection convolution.
        shortcut_conv (Conv2D): Shortcut convolutional layer.
        shortcut_bn (BatchNormalization): Batch normalization for the shortcut convolution.
        add (Add): Element-wise addition layer.
        relu (ReLU): ReLU activation layer.

    Methods:
        call(x): Perform the forward pass for the CSegBlock.

    """

    def __init__(self,*, t, filters, kernel_size, strides, out_channels, padding='same', **kwargs):
        """Initialize CSegBlock."""
        super(CSegBlock, self).__init__(**kwargs)
        total_filters = t*filters

        self.expansion_conv = tf.keras.layers.Conv2D(
            filters=total_filters, kernel_size=1, strides=2, padding=padding
        )
        self.expansion_bn = tf.keras.layers.BatchNormalization()

        self.dwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.dwise_bn = tf.keras.layers.BatchNormalization()
    
        self.upsample = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear"
        )

        self.projection_conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=1, padding=padding
        )
        self.projection_bn = tf.keras.layers.BatchNormalization()

        self.shortcut_conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=strides, padding=padding
        )
        self.shortcut_bn = tf.keras.layers.BatchNormalization()
    
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        """Forward pass for the CSegBlock"""
        
        # expansion
        expansion = self.expansion_conv(x)
        expansion = self.expansion_bn(expansion)
        expansion = self.relu(expansion)
    
        # Depthwise convolution
        dwise = self.dwise_conv(expansion)
        dwise = self.dwise_bn(dwise)
        dwise = self.relu(dwise)

        # Upsample to compensate for the downsampling in the expansion layer (stride = 2).
        # This differs from the standard MBConv.
        upsample = self.upsample(dwise)
 
        # Projection
        projection = self.projection_conv(upsample)
        projection = self.projection_bn(projection)
    
        # Shortcut
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)
        
        # residual connection
        merged = self.add([shortcut, projection])
        
        return self.relu(merged)


class ECSegBlock(tf.keras.layers.Layer):
    """Implements the extended CompSeg (ECSeg) block.

    Agrs:
        t (int): Width multiplier for filter expansion.
        filters (int): Number of input filters.
        kernel_size (int or tuple): Size of the depthwise convolution kernel.
        strides (int or tuple): Strides for the depthwise convolution.
        out_channels (int): Number of output filters for the projection convolution.
        ratio (int): Ratio for the IGCBlock instance (default is 4).
        padding (str): Padding type for convolutional layers (default is 'same').
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        expansion_conv (Conv2D): Expansion convolutional layer.
        expansion_bn (BatchNormalization): Batch normalization for the expansion convolution.
        dwise_conv (DepthwiseConv2D): Depthwise convolutional layer.
        dwise_bn (BatchNormalization): Batch normalization for the depthwise convolution.
        upsample (UpSampling2D): Upsampling layer using bilinear interpolation.
        projection_conv (Conv2D): Projection convolutional layer.
        projection_bn (BatchNormalization): Batch normalization for the projection convolution.
        shortcut_conv (Conv2D): Shortcut convolutional layer.
        shortcut_bn (BatchNormalization): Batch normalization for the shortcut convolution.
        add (Add): Element-wise addition layer.
        relu (ReLU): ReLU activation layer.
        gelu (Lambda): GELU activation layer.
        gc_modeling (IGCBlock): IGCBlock layer instance.
        
    Methods:
        call(x): Perform the forward pass for the ECSegBlock.

    """
    
    def __init__(self,*, t, filters, kernel_size, strides, out_channels, ratio=4, padding='same', **kwargs):
        """Initialize CSegBlock."""
        super(ECSegBlock, self).__init__(**kwargs)
        total_filters = t*filters

        self.expansion_conv = tf.keras.layers.Conv2D(
            filters=total_filters, kernel_size=1, strides=2, padding=padding
        )
        self.expansion_bn = tf.keras.layers.BatchNormalization()

        self.dwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.dwise_bn = tf.keras.layers.BatchNormalization()
    
        self.upsample = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear"
        )

        self.projection_conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=1, padding=padding
        )
        self.projection_bn = tf.keras.layers.BatchNormalization()

        self.shortcut_conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=strides, padding=padding
        )
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.ReLU()
        self.gelu = tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x))

        self.gc_modeling = IGCBlock(out_channels, ratio=ratio)

    def call(self, x):
        """Forward pass for the ECSegBlock"""
    
        # expansion
        expansion = self.expansion_conv(x)
        expansion = self.expansion_bn(expansion)
        expansion = self.relu(expansion)
        
        # depthwise convolution
        dwise = self.dwise_conv(expansion)
        dwise = self.dwise_bn(dwise)
        dwise = self.relu(dwise)

        # upsample
        upsample = self.upsample(dwise)

        # projection
        projection = self.projection_conv(upsample)
        projection = self.projection_bn(projection)

        # shortcut
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)

        # residual connection
        rc_out = self.add([shortcut, projection])
        rc_out = self.gelu(rc_out)
    
        # global context modeling
        gc_out = self.gc_modeling(rc_out)

        return gc_out


class NASBlock(tf.keras.layers.Layer):
    """Implements the Noise-aware stem (NAS) block

    Args:
        channels (int): Number of input-output channels.
        padding (str): Padding type for convolutions (default is 'same').
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        stem_conv (Conv2D): Stem convolutional layer.
        stem_bn (BatchNormalization): Batch normalization for the stem layer.
        dwise_conv (DepthwiseConv2D): Depthwise convolutional layer.
        dwise_ln (LayerNormalization): Layer Normalization for the depthwise convolution.
        expansion_conv (Conv2D): Expansion convolutional layer.
        projection_conv (Conv2D): Projection convolutional layer.
        add (Add): Element-wise addition layer.
        relu (ReLU): Rectified Linear Unit activation layer.
        gelu (Lambda): GELU activation layer.
        
    Methods:
        call(x, verbose=False): Perform the forward pass for the NASBlock.

    """
    
    def __init__(self, channels, padding='same', **kwargs):
        """Initialize NASBlock."""
        super(NASBlock, self).__init__(**kwargs)

        self.stem_conv = tf.keras.layers.Conv2D(
            filters=channels, kernel_size=3, strides=1, padding=padding
        )
        self.stem_bn = tf.keras.layers.BatchNormalization()

        self.dwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=7, strides=1, padding=padding
        )
        self.dwise_ln = tf.keras.layers.LayerNormalization()

        self.expansion_conv = tf.keras.layers.Conv2D(
            filters=2*channels, kernel_size=1, strides=1, padding=padding
        )

        self.projection_conv = tf.keras.layers.Conv2D(
            filters=channels, kernel_size=1, strides=1, padding=padding
        )
        
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.ReLU()
        self.gelu = tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x))
        

    def call(self, x, verbose=False):
        """Forward pass for the NASBlock"""
        
        # stem
        stem = self.stem_conv(x)
        stem = self.stem_bn(stem)
        stem = self.relu(stem)
    
        # depthwise convolution
        dwise = self.dwise_conv(stem)
        dwise = self.dwise_ln(dwise)
    
        # expansion    
        expansion = self.expansion_conv(dwise)
        expansion = self.gelu(expansion)
    
        # projection
        projection = self.projection_conv(expansion)

        return self.add([stem, projection])


class RBTBlock(tf.keras.layers.Layer):
    """Implements the Residual Bottleneck Transformer (RBT) Block.

    Args:
        t (int): Width multiplier for filter expansion.
        filters (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_trans_encoder (int): Number of Transformer encoder layers (default is 6).
        num_heads (int): Number of attention heads in the Transformer encoder (default is 8).
        d_model (int): Dimension of the MLP hidden layer in the Transformer encoder (default is 1024).
        dropout (float): Dropout rate in the Transformer encoder (default is 0.10).
        padding (str): Padding type for convolutions (default is 'same').
        **kwargs: Additional keyword arguments for the parent class.
    
    Attributes:
        dwise1_conv (DepthwiseConv2D): First depthwise convolution layer.
        dwise1_bn (BatchNormalization): Batch normalization for the first depthwise convolution.
        expansion_conv (Conv2D): Expansion convolution layer.
        expansion_bn (BatchNormalization): Batch normalization for the expansion convolution.
        reshape_in (Lambda): Reshape layer for transformer input.
        trans (list): List of TransformerEncoder layers.
        reshape_out (Lambda): Reshape layer for transformer output.
        projection_conv (Conv2D): Projection convolution layer.
        projection_bn (BatchNormalization): Batch normalization for the projection convolution.
        dwise2_conv (DepthwiseConv2D): Second depthwise convolution layer.
        dwise2_bn (BatchNormalization): Batch normalization for the second depthwise convolution.
        shortcut_conv (Conv2D): Shortcut convolution layer.
        shortcut_bn (BatchNormalization): Batch normalization for the shortcut convolution.
        relu (ReLU): ReLU activation layer.
        add (Add): Element-wise addition layer.
        
    Methods:
        call(x, verbose=False): Forward pass for the RBTBlock.

    """
    
    def __init__(self, *, t, filters, out_channels, num_trans_encoder=6, num_heads=8, d_model=1024, dropout=0.10, padding='same', **kwargs):
        """Initialize RBTBlock."""
        super(RBTBlock, self).__init__(**kwargs)
        total_filters = t*filters
    
        self.dwise1_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding=padding
        )
        self.dwise1_bn = tf.keras.layers.BatchNormalization()

        self.expansion_conv = tf.keras.layers.Conv2D(
            filters=total_filters, kernel_size=1, strides=1, padding=padding
        )
        self.expansion_bn = tf.keras.layers.BatchNormalization()

        self.reshape_in = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[-1], tf.shape(x)[1]*tf.shape(x)[2]])
        )
        self.reshape_out = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=[tf.shape(x)[0], 32, 32, tf.shape(x)[1]])
        )

        self.trans =  [TransformerEncoder(intermediate_dim=d_model, num_heads=num_heads, dropout=dropout)
                   for _ in range(num_trans_encoder)]

        self.projection_conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=1, padding='same'
        )
        self.projection_bn = tf.keras.layers.BatchNormalization()

        self.dwise2_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=2, padding=padding
        )
        self.dwise2_bn = tf.keras.layers.BatchNormalization()
        
        self.shortcut_conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, strides=2, padding='same'
        )
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()
        self.add = tf.keras.layers.Add()

    def call(self, x, verbose=False):
        """Forward pass for the RBTBlock."""
        
        # first depthwise convolution 
        dwise1 = self.dwise1_conv(x)
        dwise1 = self.dwise1_bn(dwise1)
        dwise1 = self.relu(dwise1)
    
        # expansion
        expansion = self.expansion_conv(dwise1)
        expansion = self.expansion_bn(expansion)
    
        # Reshape for transformer input
        reshaped = self.reshape_in(expansion)
    
        # transformer encoding
        for layer in self.trans:
           reshaped = layer(reshaped)

        # Reshape back
        reshaped = self.reshape_out(reshaped)
    
        # projection
        projection = self.projection_conv(reshaped)
        projection = self.projection_bn(projection)
        projection = self.relu(projection)
    
        # second depthwise convolution 
        dwise2 = self.dwise2_conv(projection)
        dwise2 = self.dwise2_bn(dwise2)

        # residual connection
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)
        shortcut = self.add([shortcut, dwise2])
    
        return self.relu(shortcut)


class UpAndConcat(tf.keras.layers.Layer):
    """Implements Upsampling and Concatenation.

    Args:
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        upsample (UpSampling2D): Upsampling layer.
        cat (Concatenate): Concatenation layer.

    Methods:
        call(inputs, verbose=False): Forward pass for the UpAndConcat layer.

    """
    
    def __init__(self, **kwargs):
        """Initialize the UpAndConcat layer."""
        super(UpAndConcat, self).__init__(**kwargs)
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.cat = tf.keras.layers.Concatenate()
  
    def call(self, inputs, verbose=False):
        """Forward pass the UpAndConcat layer."""
        x, x_skip = inputs
        upsample = self.upsample(x)
        merged = self.cat([upsample, x_skip])
        return merged 
        
        
class SegmentationHead(tf.keras.layers.Layer):
    """Implements the Segmentation Head Layer.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        num_classes (int): Number of classes in the segmentation task.
        conv_out (Conv2D): Convolutional layer for generating segmentation mask.
        activation (Activation): Activation layer (sigmoid for binary, softmax for multi-class).

    Methods:
        call(x): Forward pass for the SegmentationHead layer.

    """
    
    def __init__(self, num_classes, **kwargs):
        """Initialize SegmentationHead Layer."""
        super(SegmentationHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        self.conv_out = tf.keras.layers.Conv2D(
            filters=num_classes, kernel_size=1, strides=1, padding="same"
        )
        
        if num_classes == 1:
            self.activation = tf.keras.layers.Activation("sigmoid")
        else:
            self.activation = tf.keras.layers.Activation("softmax")

    def call(self, x):
        """Forward pass the SegmentationHead layer."""
        x = self.conv_out(x)
        x = self.activation(x)
        return x


class CompSegNet(tf.keras.Model):
    """Comprehensive Segmentation Network.

    Args:
        num_classes (int): Number of classes in the segmentation task (default is 1).
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        out_channels (list): List of output channels for each stage of the network.
        stem (NASBlock): Stem block of the network.
        encoder_block1, encoder_block2, encoder_block3 (Sequential): Encoder blocks for different stages.
        bottleneck_block (RBTBlock): Bottleneck block in the middle of the network.
        decoder_block1, decoder_block2, decoder_block3, decoder_block4 (ECSegBlock): Decoder blocks for different stages.
        up_and_concat (UpAndConcat): Upsampling and concatenation layer for decoder stages.
        seg (SegmentationHead): Segmentation head for mask generation.

    Methods:
        call(x, verbose=False): Forward pass for the CompSegNet.
        build_graph(input_shape): Build the model graph with specified input shape.
        
    """
    
    def __init__(self, num_classes=1, **kwargs):
        super(CompSegNet, self).__init__(**kwargs)
        self.out_channels = [64, 128, 256, 512, 1024]
        
        self.stem = NASBlock(channels=self.out_channels[0])

        self.encoder_block1 = tf.keras.Sequential([
            CSegBlock(t=4, filters=self.out_channels[0], kernel_size=7, strides=1, out_channels=self.out_channels[1]),
            ECSegBlock(t=4, filters=self.out_channels[1], kernel_size=7, strides=2, out_channels=self.out_channels[1])
        ])
        self.encoder_block2 = tf.keras.Sequential([
            CSegBlock(t=4, filters=self.out_channels[1], kernel_size=7, strides=1, out_channels=self.out_channels[2]),
            ECSegBlock(t=4, filters=self.out_channels[2], kernel_size=7, strides=2, out_channels=self.out_channels[2])
        ])
        self.encoder_block3 = tf.keras.Sequential([
            CSegBlock(t=4, filters=self.out_channels[2], kernel_size=7, strides=1, out_channels=self.out_channels[3]),
            ECSegBlock(t=4, filters=self.out_channels[3], kernel_size=7, strides=2, out_channels=self.out_channels[3])
        ])

        self.bottleneck_block = RBTBlock(t=4, filters=self.out_channels[3], out_channels=self.out_channels[4])
        
        self.decoder_block1 = ECSegBlock(
            t=4, filters=self.out_channels[4], kernel_size=7, strides=1, out_channels=self.out_channels[3]
        )
        self.decoder_block2 = ECSegBlock(
            t=4, filters=self.out_channels[3], kernel_size=7, strides=1, out_channels=self.out_channels[2]
        )
        self.decoder_block3 = ECSegBlock(
            t=4, filters=self.out_channels[2], kernel_size=7, strides=1, out_channels=self.out_channels[1]
        )
        self.decoder_block4 = ECSegBlock(
            t=4, filters=self.out_channels[1], kernel_size=7, strides=1, out_channels=self.out_channels[0]
        )
        
        self.up_and_concat = UpAndConcat()
         
        self.seg = SegmentationHead(num_classes=1)
        
    def call(self, x, verbose=False):
        """Forward pass for the CompSegNet model"""
        
        # stem
        stem = self.stem(x)
        
        # encoder path
        enc1 = self.encoder_block1(stem)
        enc2 = self.encoder_block2(enc1)
        enc3 = self.encoder_block3(enc2)
        
        # bottleneck
        bneck = self.bottleneck_block(enc3)    
        
        # decoder path
        uc1 = self.up_and_concat([bneck, enc3])
        dec1 = self.decoder_block1(uc1)
        uc2 = self.up_and_concat([dec1, enc2])
        dec2 = self.decoder_block2(uc2)
        uc3 = self.up_and_concat([dec2, enc1])
        dec3 = self.decoder_block3(uc3)
        uc4 = self.up_and_concat([dec3, stem])
        dec4 = self.decoder_block4(uc4)
        
        # segmentation mask generation
        seg_mask = self.seg(dec4) 
        
        return seg_mask
               
    def build_graph(self, input_shape):
        """Build the CompSegNet model graph."""
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

if __name__ == "__main__":
   input_shape = (256, 256, 3)
   
   model = CompSegNet()
   model = model.build_graph(input_shape)
   model.summary()

