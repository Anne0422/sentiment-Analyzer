<?php
/**
 * Plugin Name: AI Sentiment Comment Tracker
 * Description: Sends comments to a Python Flask API for sentiment analysis.
 */

add_action('comment_post', 'send_comment_to_ai', 10, 3);

function send_comment_to_ai($comment_ID, $comment_approved, $commentdata) {
    $api_url = 'http://YOUR_SERVER_IP:5000/analyze'; // Replace with your Flask URL
    
    $payload = array(
        'comment_id' => $comment_ID,
        'text'       => $commentdata['comment_content']
    );

    $response = wp_remote_post($api_url, array(
        'method'    => 'POST',
        'body'      => json_encode($payload),
        'headers'   => array('Content-Type' => 'application/json'),
    ));

    if (!is_wp_error($response)) {
        $result = json_decode(wp_remote_retrieve_body($response), true);
        // Store the AI result in comment meta
        update_comment_meta($comment_ID, 'sentiment_score', $result['score']);
        update_comment_meta($comment_ID, 'sentiment_label', $result['label']);
    }
}