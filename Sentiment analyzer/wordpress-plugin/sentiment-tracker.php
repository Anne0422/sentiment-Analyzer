<?php
/**
 * Plugin Name: AI Sentiment Comment Tracker
 * Description: Sends comments to a Python Flask API and displays sentiment analysis in the dashboard.
 * Version: 1.0
 * Author: Anne
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * 1. THE BRIDGE: Send comment to Flask API
 */
add_action('comment_post', 'send_comment_to_python_api', 10, 3);

function send_comment_to_python_api($comment_ID, $comment_approved, $commentdata) {
    // CHANGE THIS: Use http://localhost:5000/analyze for local or your Client's IP for remote
    $api_url = 'http://127.0.0.1:5000/analyze'; 

    $payload = array(
        'comment_id' => $comment_ID,
        'text'       => $commentdata['comment_content']
    );

    $response = wp_remote_post($api_url, array(
        'method'    => 'POST',
        'body'      => json_encode($payload),
        'headers'   => array('Content-Type' => 'application/json'),
        'timeout'   => 15 // AI needs a moment to think
    ));

    if (!is_wp_error($response)) {
        $body = json_decode(wp_remote_retrieve_body($response), true);
        
        // 2. THE STORAGE: Save results to database
        if (isset($body['label'])) {
            update_comment_meta($comment_ID, 'sentiment_label', $body['label']);
            update_comment_meta($comment_ID, 'sentiment_score', $body['score']);
        }
    }
}

/**
 * 3. THE UI: Add a column in the WordPress Comments table
 */
add_filter('manage_edit-comments_columns', 'add_sentiment_column');
function add_sentiment_column($columns) {
    $columns['sentiment'] = 'AI Sentiment';
    return $columns;
}

add_action('manage_comments_custom_column', 'display_sentiment_column', 10, 2);
function display_sentiment_column($column, $comment_ID) {
    if ($column == 'sentiment') {
        $label = get_comment_meta($comment_ID, 'sentiment_label', true);
        $score = get_comment_meta($comment_ID, 'sentiment_score', true);

        if ($label) {
            $color = ($label == 'Positive') ? 'green' : (($label == 'Negative') ? 'red' : 'orange');
            echo "<strong style='color: $color;'>$label</strong><br/><small>Score: $score</small>";
        } else {
            echo "<span style='color: gray;'>Pending...</span>";
        }
    }
}
